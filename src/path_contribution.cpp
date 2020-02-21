#include "path_contribution.h"
#include "medium.h"
#include "parallel.h"
#include "scene.h"

// Evaluate transmittance between two points, with potentially an
// intermediate surface between them.
DEVICE
inline
Vector3 eval_transmittance(const FlattenScene &scene,
                           const Intersection &int_isect,
                           const Ray &org_ray,
                           const Ray &int_ray,
                           const int medium_id,
                           const int int_medium_id,
                           const SurfacePoint &int_point,
                           const Intersection &tgt_isect,
                           const SurfacePoint &tgt_point) {
    if (!tgt_isect.valid()) {
        return Vector3{0, 0, 0};
    }
    if (int_isect.valid()) {
        const auto &shape = scene.shapes[int_isect.shape_id];
        if (int_isect == tgt_isect) {
            // There is no intermediate intersect
            if (medium_id != -1) {
                const auto &medium = scene.mediums[medium_id];
                auto dist = distance(org_ray.org,
                                     tgt_point.position);
                return transmittance(medium, dist);
            } else {
                return Vector3{1, 1, 1};
            }
        } else {
            if (shape.material_id != -1) {
                // Hit opaque object, transmittance = 0
                return Vector3{0, 0, 0};
            } else {
                // Hit media bounary
                // tmax <= 0 means the ray is blocked
                if (int_ray.tmax > 0) {
                    auto tr0 = Vector3{1, 1, 1};
                    auto tr1 = Vector3{1, 1, 1};
                    if (medium_id != -1) {
                        const auto &medium = scene.mediums[medium_id];
                        auto dist = distance(org_ray.org, int_point.position);
                        tr0 = transmittance(medium, dist);
                    }
                    if (int_medium_id != -1) {
                        const auto &medium = scene.mediums[int_medium_id];
                        auto dist = distance(int_point.position,
                                             tgt_point.position);
                        tr1 = transmittance(medium, dist);
                    }
                    return tr0 * tr1;
                } else {
                    // Block by something
                    return Vector3{0, 0, 0};
                }
            }
        }
    } else {
        if (medium_id != -1) {
            const auto &medium = scene.mediums[medium_id];
            auto dist = distance(org_ray.org,
                                 tgt_point.position);
            return transmittance(medium, dist);
        } else {
            // vaccum
            return Vector3{1, 1, 1};
        }
    }
}

struct path_contribs_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &throughput = throughputs[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        const auto &surface_isect = surface_isects[pixel_id];
        const auto &surface_point = surface_points[pixel_id];
        const auto &light_isect = light_isects[pixel_id];
        const auto &light_point = light_points[pixel_id];
        const auto &light_ray = light_rays[pixel_id];
        const auto &scatter_isect = scatter_isects[pixel_id];
        const auto &scatter_point = scatter_points[pixel_id];
        const auto &scatter_ray = scatter_rays[pixel_id];
        const auto &min_rough = min_roughness[pixel_id];
        auto &next_throughput = next_throughputs[pixel_id];
        next_throughput = Vector3{0, 0, 0};

        auto wi = -incoming_ray.dir;
        auto p = surface_point.position;
        if (medium_ids != nullptr && medium_distances != nullptr) {
            if (medium_ids[pixel_id] >= 0) {
                p = incoming_ray.org + incoming_ray.dir * medium_distances[pixel_id];
            }
        }

        // Determine whether we should evaluate bsdf or phase function
        // If we are on a surface, and the surface has material assigned,
        // we should evaluate bsdf
        // If we are not on a surface, we should evaluate phase function
        // If we are on a surface, and the surface has no material assigned,
        // we ignore the surface by doing nothing.
        auto eval_phase = false;
        if (medium_ids != nullptr && medium_distances != nullptr &&
              medium_ids[pixel_id] >= 0 &&
              medium_distances[pixel_id] < incoming_ray.tmax) {
            eval_phase = true;
        }

        // Next event estimation
        auto nee_contrib = Vector3{0, 0, 0};
        if (light_ray.tmax >= 0) { // tmax < 0 means the ray is blocked
            if (light_isect.valid()) {
                // area light
                const auto &light_shape = scene.shapes[light_isect.shape_id];
                auto dir = light_point.position - p;
                auto dist_sq = length_squared(dir);
                auto wo = dir / sqrt(dist_sq);
                if (dist_sq > 1e-20f && light_shape.light_id >= 0) {
                    const auto &light = scene.area_lights[light_shape.light_id];
                    if (light.two_sided || dot(-wo, light_point.shading_frame.n) > 0) {
                        auto geometry_term = fabs(dot(wo, light_point.geom_normal)) / dist_sq;
                        auto light_contrib = light.intensity;
                        auto light_pmf = scene.light_pmf[light_shape.light_id];
                        auto light_area = scene.light_areas[light_shape.light_id];
                        auto pdf_nee = light_pmf / light_area;
                        if (eval_phase) {
                            auto phase_function =
                                get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                            auto phase_val = phase_function_eval(
                                phase_function, wi, wo);
                            auto pdf_phase = phase_function_pdf(
                                phase_function, wi, wo) * geometry_term;
                            auto mis_weight = Real(1 / (1 + square((double)pdf_phase / (double)pdf_nee)));
                            nee_contrib =
                                (mis_weight * geometry_term / pdf_nee) * phase_val * light_contrib;
                        } else {
                            // On a surface
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            if (surface_shape.material_id >= 0) {
                                const auto &material = scene.materials[surface_shape.material_id];
                                auto bsdf_val = bsdf(material, surface_point, wi, wo, min_rough);
                                auto pdf_bsdf =
                                    bsdf_pdf(material, surface_point, wi, wo, min_rough) * geometry_term;

                                auto mis_weight = Real(1 / (1 + square((double)pdf_bsdf / (double)pdf_nee)));
                                nee_contrib =
                                    (mis_weight * geometry_term / pdf_nee) * bsdf_val * light_contrib;
                            }
                        }
                    }
                }
            } else if (scene.envmap != nullptr) {
                // Environment light
                auto wo = light_ray.dir;
                auto envmap_id = scene.num_lights - 1;
                auto light_pmf = scene.light_pmf[envmap_id];
                auto pdf_nee = envmap_pdf(*scene.envmap, wo) * light_pmf;
                if (pdf_nee > 0) {
                    // XXX: For now we don't use ray differentials for envmap
                    //      A proper approach might be to use a filter radius based on sampling density?
                    RayDifferential ray_diff{Vector3{0, 0, 0}, Vector3{0, 0, 0},
                                             Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                    auto light_contrib = envmap_eval(*scene.envmap, wo, ray_diff);
                    if (eval_phase) {
                        auto phase_function =
                            get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                        auto phase_val = phase_function_eval(
                            phase_function, wi, wo);
                        auto pdf_phase = phase_function_pdf(
                            phase_function, wi, wo);
                        auto mis_weight = Real(1 / (1 + square((double)pdf_phase / (double)pdf_nee)));
                        nee_contrib = (mis_weight / pdf_nee) * phase_val * light_contrib;
                    } else {
                        const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                        if (surface_shape.material_id >= 0) {
                            const auto &material = scene.materials[surface_shape.material_id];
                            auto bsdf_val = bsdf(material, surface_point, wi, wo, min_rough);
                            auto pdf_bsdf = bsdf_pdf(material, surface_point, wi, wo, min_rough);
                            auto mis_weight = Real(1 / (1 + square((double)pdf_bsdf / (double)pdf_nee)));
                            nee_contrib = (mis_weight / pdf_nee) * bsdf_val * light_contrib;
                        }
                    }
                }
            }
            // Transmittance
            if (medium_ids != nullptr) {
                nee_contrib *= eval_transmittance(scene,
                                                  light_int_isects[pixel_id],
                                                  light_ray,
                                                  light_int_rays[pixel_id],
                                                  medium_ids[pixel_id],
                                                  light_medium_ids[pixel_id],
                                                  light_int_points[pixel_id],
                                                  light_isect,
                                                  light_point);
            }
        }

        // BSDF or phase function importance sampling.
        // Also update throughput.
        auto scatter_contrib = Vector3{0, 0, 0};
        if (scatter_isect.valid()) { // We hit a surface
            const auto &light_shape = scene.shapes[scatter_isect.shape_id];
            if (length_squared(scatter_point.position - p) > 1e-20f) {
                auto wo = scatter_ray.dir;
                auto scatter_val = Vector3{1, 1, 1};
                auto scatter_pdf = Real(1);
                if (eval_phase) {
                    // Inside a medium
                    auto phase_function =
                        get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                    scatter_pdf = phase_function_pdf(phase_function, wi, wo);
                    auto phase_eval = phase_function_eval(phase_function, wi, wo);
                    // Perfect importance sampling
                    scatter_val = Vector3{phase_eval, phase_eval, phase_eval};
                } else {
                    // On a surface
                    const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                    if (surface_shape.material_id >= 0) {
                        const auto &material = scene.materials[surface_shape.material_id];
                        scatter_pdf = bsdf_pdf(material, surface_point, wi, wo, min_rough);
                        scatter_val = bsdf(material, surface_point, wi, wo, min_rough);
                    }
                }
                if (scatter_pdf > 1e-20f) {
                    if (light_shape.light_id >= 0) {
                        const auto &light = scene.area_lights[light_shape.light_id];
                        if (light.two_sided || dot(-wo, scatter_point.shading_frame.n) > 0) {
                            auto light_contrib = light.intensity;
                            auto light_pmf = scene.light_pmf[light_shape.light_id];
                            auto light_area = scene.light_areas[light_shape.light_id];
                            auto inv_area = 1 / light_area;
                            auto dist_sq = length_squared(scatter_point.position - p);
                            auto geometry_term = fabs(dot(wo, light_point.geom_normal)) / dist_sq;
                            auto pdf_nee = (light_pmf * inv_area) / geometry_term;
                            auto mis_weight = Real(1 / (1 + square((double)pdf_nee / (double)scatter_pdf)));
                            scatter_contrib = (mis_weight / scatter_pdf) * scatter_val * light_contrib;
                        }
                    }
                    // Update throughput
                    next_throughput = throughput * (scatter_val / scatter_pdf);
                    if (path_transmittances != nullptr) {
                        next_throughput *= path_transmittances[pixel_id];
                    }
                }
            }
        } else {
            // Hit nothing
            auto wo = scatter_ray.dir;
            // wo can be zero when bsdf_sample failed
            if (length_squared(wo) > 0) {
                auto scatter_val = Vector3{1, 1, 1};
                auto scatter_pdf = Real(1);
                if (eval_phase) {
                    // Inside a medium
                    auto phase_function =
                        get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                    scatter_pdf = phase_function_pdf(phase_function, wi, wo);
                    auto phase_eval = phase_function_eval(phase_function, wi, wo);
                    scatter_val = Vector3{phase_eval, phase_eval, phase_eval};
                } else {
                    // On a surface
                    const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                    if (surface_shape.material_id >= 0) {
                        const auto &material = scene.materials[surface_shape.material_id];
                        scatter_pdf = bsdf_pdf(material, surface_point, wi, wo, min_rough);
                        scatter_val = bsdf(material, surface_point, wi, wo, min_rough);
                    }
                }
                if (scene.envmap != nullptr && scatter_pdf > 1e-20f) {
                    // Hit an environment map
                    // XXX: For now we don't use ray differentials for envmap
                    //      A proper approach might be to use a filter radius based on sampling density?
                    RayDifferential ray_diff{Vector3{0, 0, 0}, Vector3{0, 0, 0},
                                             Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                    auto light_contrib = envmap_eval(*scene.envmap, wo, ray_diff);
                    auto envmap_id = scene.num_lights - 1;
                    auto light_pmf = scene.light_pmf[envmap_id];
                    auto pdf_nee = envmap_pdf(*scene.envmap, wo) * light_pmf;
                    auto mis_weight = Real(1 / (1 + square((double)pdf_nee / (double)scatter_pdf)));
                    scatter_contrib = (mis_weight / scatter_pdf) * scatter_val * light_contrib;
                    next_throughput = throughput * (scatter_val / scatter_pdf);
                    if (path_transmittances != nullptr) {
                        next_throughput *= path_transmittances[pixel_id];
                    }
                }
            }
        }
        // Transmittance
        if (medium_ids != nullptr) {
            scatter_contrib *= eval_transmittance(scene,
                                                  next_isects[pixel_id],
                                                  scatter_ray,
                                                  scatter_int_rays[pixel_id],
                                                  medium_ids[pixel_id],
                                                  scatter_medium_ids[pixel_id],
                                                  next_points[pixel_id],
                                                  scatter_isect,
                                                  scatter_point);
        }

        assert(isfinite(nee_contrib));
        assert(isfinite(scatter_contrib));
        auto path_contrib = throughput * (nee_contrib + scatter_contrib);
        if (path_transmittances != nullptr) {
            path_contrib *= path_transmittances[pixel_id];
        }
        if (rendered_image != nullptr) {
            auto nd = channel_info.num_total_dimensions;
            auto d = channel_info.radiance_dimension;
            rendered_image[nd * pixel_id + d    ] += weight * path_contrib[0];
            rendered_image[nd * pixel_id + d + 1] += weight * path_contrib[1];
            rendered_image[nd * pixel_id + d + 2] += weight * path_contrib[2];
        }
        if (edge_contribs != nullptr) {
            edge_contribs[pixel_id] += sum(weight * path_contrib);
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Vector3 *throughputs;
    const Vector3 *path_transmittances;
    // Information of the shading point
    const Ray *incoming_rays;
    const Intersection *surface_isects;
    const SurfacePoint *surface_points;
    const int *medium_ids;
    const Real *medium_distances;
    // Information of next event estimation
    const Intersection *light_isects;
    const SurfacePoint *light_points;
    const Ray *light_rays;
    const Intersection *light_int_isects;
    const SurfacePoint *light_int_points;
    const Ray *light_int_rays;
    const int *light_medium_ids;
    // Information of scattering
    const Intersection *scatter_isects;
    const SurfacePoint *scatter_points;
    const Ray *scatter_rays;
    const Intersection *next_isects;
    const SurfacePoint *next_points;
    const Ray *scatter_int_rays;
    const int *scatter_medium_ids;
    // Misc
    const Real *min_roughness;
    const Real weight;
    const ChannelInfo channel_info;
    // Output
    Vector3 *next_throughputs;
    float *rendered_image;
    Real *edge_contribs;
};

struct d_path_contribs_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &throughput = throughputs[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        const auto &surface_isect = surface_isects[pixel_id];
        const auto &surface_point = surface_points[pixel_id];
        const auto &light_isect = light_isects[pixel_id];
        const auto &light_ray = light_rays[pixel_id];
        const auto &directional_ray = directional_rays[pixel_id];
        const auto &min_rough = min_roughness[pixel_id];
        const auto &directional_point = directional_points[pixel_id];
        const auto &d_next_throughput = d_next_throughputs[pixel_id];

        auto &d_throughput = d_throughputs[pixel_id];
        auto &d_incoming_ray = d_incoming_rays[pixel_id];
        auto &d_shading_point = d_shading_points[pixel_id];
        auto &d_next_ray = d_next_rays[pixel_id];

        auto wi = -incoming_ray.dir;
        auto p = surface_point.position;
        if (medium_ids != nullptr && medium_distances != nullptr) {
            if (medium_ids[pixel_id] > 0) {
                p = p + incoming_ray.dir * medium_distances[pixel_id];
            }
        }

        // Determine whether we should evaluate bsdf or phase function
        // If we are on a surface, and the surface has material assigned,
        // we should evaluate bsdf
        // If we are not on a surface, we should evaluate phase function
        // If we are on a surface, and the surface has no material assigned,
        // we ignore the surface by doing nothing.
        auto eval_phase = false;
        if (medium_ids != nullptr && medium_distances != nullptr &&
              medium_ids[pixel_id] >= 0 &&
              medium_distances[pixel_id] < incoming_ray.tmax) {
            eval_phase = true;
        }

        const auto &shading_shape = scene.shapes[surface_isect.shape_id];
        const auto &material = scene.materials[shading_shape.material_id];

        auto &d_material = d_materials[shading_shape.material_id];

        auto nd = channel_info.num_total_dimensions;
        auto d = channel_info.radiance_dimension;
        // rendered_image[nd * pixel_id + d    ] += weight * path_contrib[0];
        // rendered_image[nd * pixel_id + d + 1] += weight * path_contrib[1];
        // rendered_image[nd * pixel_id + d + 2] += weight * path_contrib[2];
        auto d_path_contrib = weight *
            Vector3{d_rendered_image[nd * pixel_id + d    ],
                    d_rendered_image[nd * pixel_id + d + 1],
                    d_rendered_image[nd * pixel_id + d + 2]};

        // Initialize derivatives
        d_throughput = Vector3{0, 0, 0};
        d_incoming_ray = DRay{};
        d_shading_point = SurfacePoint::zero();
        if (d_path_transmittances != nullptr) {
            d_path_transmittances[pixel_id] = Vector3{0, 0, 0};
        }

        auto path_transmittance = Vector3{1, 1, 1};
        if (path_transmittances != nullptr) {
            path_transmittance = path_transmittances[pixel_id];
        }

        // Next event estimation
        if (light_ray.tmax >= 0) { // tmax < 0 means the ray is blocked
            if (light_isect.valid()) {
                // Area light
                const auto &light_shape = scene.shapes[light_isect.shape_id];
                const auto &light_sample = light_samples[pixel_id];
                const auto &light_point = light_points[pixel_id];

                auto dir = light_point.position - p;
                auto dist_sq = length_squared(dir);
                auto wo = dir / sqrt(dist_sq);
                if (light_shape.light_id >= 0) {
                    const auto &light = scene.area_lights[light_shape.light_id];
                    if (light.two_sided || dot(-wo, light_point.shading_frame.n) > 0) {
                        Vector3 d_light_vertices[3] = {
                            Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                        auto cos_light = dot(wo, light_point.geom_normal);
                        auto geometry_term = fabs(cos_light) / dist_sq;
                        const auto &light = scene.area_lights[light_shape.light_id];
                        auto light_contrib = light.intensity;
                        auto light_pmf = scene.light_pmf[light_shape.light_id];
                        auto light_area = scene.light_areas[light_shape.light_id];
                        auto inv_area = 1 / light_area;
                        auto pdf_nee = light_pmf * inv_area;

                        if (eval_phase) {
                            // Compute the phase function pdf
                            auto phase_function =
                                get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                            auto phase_val = phase_function_eval(
                                phase_function, wi, wo);
                            auto pdf_phase = phase_function_pdf(
                                phase_function, wi, wo) * geometry_term;
                            auto mis_weight = Real(1 / (1 + square((double)pdf_phase / (double)pdf_nee)));
                            auto nee_contrib =
                                (mis_weight * geometry_term / pdf_nee) *
                                light_contrib * phase_val;

                            // path_contrib = path_transmittance * throughput * (nee_contrib + scatter_contrib)
                            auto d_nee_contrib = d_path_contrib * throughput * path_transmittance;
                            d_throughput += d_path_contrib * nee_contrib * path_transmittance;
                            if (d_path_transmittances != nullptr) {
                                d_path_transmittances[pixel_id] += d_path_contrib * throughput * nee_contrib;
                            }

                            auto weight = mis_weight / pdf_nee;
                            // nee_contrib = (weight * geometry_term) *
                            //                phase_val * light_contrib
                            // Ignore derivatives of MIS & PMF
                            auto d_geometry_term =
                                weight * phase_val * sum(d_nee_contrib * light_contrib);
                            auto d_light_contrib =
                                weight * d_nee_contrib * phase_val * geometry_term;
                            auto d_weight = geometry_term * phase_val *
                                sum(d_nee_contrib * light_contrib);
                            auto d_phase_val = (weight * geometry_term * phase_val) *
                                sum(d_nee_contrib * light_contrib);

                            // phase_val = phase_function_eval(phase_function, wi, wo)
                            auto d_phase_function = DPhaseFunction(phase_function);
                            auto d_wi = Vector3{0, 0, 0};
                            auto d_wo = Vector3{0, 0, 0};
                            d_phase_function_eval(phase_function, wi, wo, d_phase_val,
                                d_phase_function, d_wi, d_wo);
                            // phase_function =
                            //     get_phase_function(scene.mediums[medium_ids[pixel_id]])
                            d_get_phase_function(scene.mediums[medium_ids[pixel_id]],
                                                 d_phase_function,
                                                 d_mediums[pixel_id]);

                            // weight = mis_weight / pdf_nee
                            auto d_pdf_nee = -d_weight * weight / pdf_nee;
                            // pdf_nee = light_pmf / light_area
                            //         = light_pmf * tri_pmf / tri_area
                            auto d_area =
                                -d_pdf_nee * pdf_nee / get_area(light_shape, light_isect.tri_id);
                            d_get_area(light_shape, light_isect.tri_id, d_area, d_light_vertices);

                            // light_contrib = light.intensity
                            atomic_add(d_area_lights[light_shape.light_id].intensity, d_light_contrib);

                            // geometry_term = fabs(cos_light) / dist_sq
                            auto d_cos_light = cos_light > 0 ?
                                d_geometry_term / dist_sq : -d_geometry_term / dist_sq;
                            auto d_dist_sq = -d_geometry_term * geometry_term / dist_sq;

                            // cos_light = dot(wo, light_point.geom_normal)
                            d_wo += d_cos_light * light_point.geom_normal;
                            auto d_light_point = SurfacePoint::zero();
                            d_light_point.geom_normal = d_cos_light * wo;

                            // wo = dir / sqrt(dist_sq)
                            auto d_dir = d_wo / sqrt(dist_sq);
                            // sqrt(dist_sq)
                            auto d_sqrt_dist_sq = -sum(d_wo * dir) / dist_sq;
                            d_dist_sq += (0.5f * d_sqrt_dist_sq / sqrt(dist_sq));
                            // dist_sq = length_squared(dir)
                            d_dir += d_length_squared(dir, d_dist_sq);
                            // dir = light_point.position - p
                            d_light_point.position += d_dir;
                            d_shading_point.position -= d_dir;
                            // wi = -incoming_ray.dir
                            d_incoming_ray.dir -= d_wi;

                            // Need to backpropagate to shape by sampling point on light
                            d_sample_shape(light_shape, light_isect.tri_id,
                                light_sample.uv, d_light_point, d_light_vertices);

                            // Accumulate light derivatives
                            auto light_tri_index = get_indices(light_shape, light_isect.tri_id);
                            atomic_add(&d_shapes[light_isect.shape_id].vertices[3 * light_tri_index[0]],
                                d_light_vertices[0]);
                            atomic_add(&d_shapes[light_isect.shape_id].vertices[3 * light_tri_index[1]],
                                d_light_vertices[1]);
                            atomic_add(&d_shapes[light_isect.shape_id].vertices[3 * light_tri_index[2]],
                                d_light_vertices[2]);
                        } else {
                            // Not in a medium

                            // Compute the BSDF pdf and everything associated with it
                            auto bsdf_val = bsdf(material, surface_point, wi, wo, min_rough);
                            auto pdf_bsdf =
                                bsdf_pdf(material, surface_point, wi, wo, min_rough) * geometry_term;
                            auto mis_weight = Real(1 / (1 + square((double)pdf_bsdf / (double)pdf_nee)));
                            auto nee_contrib = (mis_weight * geometry_term / pdf_nee) *
                                               bsdf_val * light_contrib;

                            // path_contrib = path_transmittance * throughput * (nee_contrib + scatter_contrib)
                            auto d_nee_contrib = d_path_contrib * throughput * path_transmittance;
                            d_throughput += d_path_contrib * nee_contrib * path_transmittance;
                            if (d_path_transmittances != nullptr) {
                                d_path_transmittances[pixel_id] += d_path_contrib * throughput * nee_contrib;
                            }

                            auto weight = mis_weight / pdf_nee;
                            // nee_contrib = (weight * geometry_term)
                            //                bsdf_val * light_contrib
                            // Ignore derivatives of MIS weight & PMF
                            auto d_weight = geometry_term *
                                sum(d_nee_contrib * bsdf_val * light_contrib);
                            auto d_geometry_term =
                                weight * sum(d_nee_contrib * bsdf_val * light_contrib);
                            auto d_bsdf_val =
                                weight * d_nee_contrib * geometry_term * light_contrib;
                            auto d_light_contrib =
                                weight * d_nee_contrib * geometry_term * bsdf_val;

                            // bsdf_val = bsdf(material, surface_point, wi, wo, min_rough)
                            auto d_wo = Vector3{0, 0, 0};
                            auto d_wi = Vector3{0, 0, 0};
                            d_bsdf(material, surface_point, wi, wo, min_rough, d_bsdf_val,
                                d_material, d_shading_point, d_wi, d_wo);

                            // weight = mis_weight / pdf_nee
                            auto d_pdf_nee = -d_weight * weight / pdf_nee;
                            // pdf_nee = light_pmf / light_area
                            //         = light_pmf * tri_pmf / tri_area
                            auto d_area =
                                -d_pdf_nee * pdf_nee / get_area(light_shape, light_isect.tri_id);
                            d_get_area(light_shape, light_isect.tri_id, d_area, d_light_vertices);
                            // light_contrib = light.intensity
                            atomic_add(d_area_lights[light_shape.light_id].intensity, d_light_contrib);
                            // geometry_term = fabs(cos_light) / dist_sq
                            auto d_cos_light = cos_light > 0 ?
                                d_geometry_term / dist_sq : -d_geometry_term / dist_sq;
                            auto d_dist_sq = -d_geometry_term * geometry_term / dist_sq;
                            // cos_light = dot(wo, light_point.geom_normal)
                            d_wo += d_cos_light * light_point.geom_normal;
                            auto d_light_point = SurfacePoint::zero();
                            d_light_point.geom_normal = d_cos_light * wo;
                            // wo = dir / sqrt(dist_sq)
                            auto d_dir = d_wo / sqrt(dist_sq);
                            // sqrt(dist_sq)
                            auto d_sqrt_dist_sq = -sum(d_wo * dir) / dist_sq;
                            d_dist_sq += (0.5f * d_sqrt_dist_sq / sqrt(dist_sq));
                            // dist_sq = length_squared(dir)
                            d_dir += d_length_squared(dir, d_dist_sq);
                            // dir = light_point.position - p
                            d_light_point.position += d_dir;
                            d_shading_point.position -= d_dir;
                            // wi = -incoming_ray.dir
                            d_incoming_ray.dir -= d_wi;

                            // sample point on light
                            d_sample_shape(light_shape, light_isect.tri_id,
                                light_sample.uv, d_light_point, d_light_vertices);

                            // Accumulate derivatives
                            auto light_tri_index = get_indices(light_shape, light_isect.tri_id);
                            atomic_add(&d_shapes[light_isect.shape_id].vertices[3 * light_tri_index[0]],
                                d_light_vertices[0]);
                            atomic_add(&d_shapes[light_isect.shape_id].vertices[3 * light_tri_index[1]],
                                d_light_vertices[1]);
                            atomic_add(&d_shapes[light_isect.shape_id].vertices[3 * light_tri_index[2]],
                                d_light_vertices[2]);
                        }
                    }
                }
            } else if (scene.envmap != nullptr) {
                // Environment light
                auto wo = light_ray.dir;
                auto envmap_id = scene.num_lights - 1;
                auto light_pmf = scene.light_pmf[envmap_id];
                auto pdf_nee = envmap_pdf(*scene.envmap, wo) * light_pmf;
                if (pdf_nee > 0) {
                    // XXX: For now we don't use ray differentials for next event estimation.
                    //      A proper approach might be to use a filter radius based on sampling density?
                    auto ray_diff = RayDifferential{
                        Vector3{0, 0, 0}, Vector3{0, 0, 0},
                        Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                    auto light_contrib = envmap_eval(*scene.envmap, wo, ray_diff);
                    if (eval_phase) {
                        auto phase_function =
                            get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                        auto phase_val = phase_function_eval(phase_function, wi, wo);
                        auto pdf_phase = phase_function_pdf(phase_function, wi, wo);
                        auto mis_weight = Real(1 / (1 + square((double)pdf_phase / (double)pdf_nee)));
                        auto nee_contrib = (mis_weight / pdf_nee) * light_contrib * phase_val;

                        // path_contrib = path_transmittance * throughput * (nee_contrib + scatter_contrib)
                        auto d_nee_contrib = d_path_contrib * throughput * path_transmittance;
                        d_throughput += d_path_contrib * nee_contrib * path_transmittance;
                        if (d_path_transmittances != nullptr) {
                            d_path_transmittances[pixel_id] += d_path_contrib * throughput * nee_contrib;
                        }

                        auto weight = mis_weight / pdf_nee;

                        // nee_contrib = weight * light_contrib * phase_val
                        auto d_light_contrib = weight * d_nee_contrib * phase_val;
                        auto d_phase_val = weight * sum(d_nee_contrib);

                        // phase_val = phase_function_eval(phase_function, wi, wo)
                        auto d_phase_function = DPhaseFunction(phase_function);
                        auto d_wi = Vector3{0, 0, 0};
                        auto d_wo = Vector3{0, 0, 0};
                        d_phase_function_eval(phase_function, wi, wo, d_phase_val,
                            d_phase_function, d_wi, d_wo);
                        // phase_function =
                        //     get_phase_function(scene.mediums[medium_ids[pixel_id]])
                        d_get_phase_function(scene.mediums[medium_ids[pixel_id]],
                                             d_phase_function,
                                             d_mediums[pixel_id]);

                        auto d_ray_diff = RayDifferential{
                            Vector3{0, 0, 0}, Vector3{0, 0, 0},
                            Vector3{0, 0, 0}, Vector3{0, 0, 0}};

                        // light_contrib = eval_envmap(*scene.envmap, wo, ray_diff)
                        d_envmap_eval(*scene.envmap, wo, ray_diff, d_light_contrib,
                            *d_envmap, d_wo, d_ray_diff);

                        // We don't propagate to pdf_nee: envmap_pdf derivatives can contain high variance

                        // wi = -incoming_ray.dir;
                        d_incoming_ray.dir -= d_wi;
                    } else {
                        auto bsdf_val = bsdf(material, surface_point, wi, wo, min_rough);
                        auto pdf_bsdf = bsdf_pdf(material, surface_point, wi, wo, min_rough);
                        auto mis_weight = Real(1 / (1 + square((double)pdf_bsdf / (double)pdf_nee)));
                        auto nee_contrib = (mis_weight / pdf_nee) * bsdf_val * light_contrib;

                        // path_contrib = path_transmittance * throughput * (nee_contrib + scatter_contrib)
                        auto d_nee_contrib = d_path_contrib * throughput * path_transmittance;
                        d_throughput += d_path_contrib * nee_contrib * path_transmittance;
                        if (d_path_transmittances != nullptr) {
                            d_path_transmittances[pixel_id] += d_path_contrib * throughput * nee_contrib;
                        }

                        auto weight = mis_weight / pdf_nee;
                        // nee_contrib = weight * bsdf_val * light_contribe
                        // Ignore derivatives of MIS weight & pdf
                        auto d_bsdf_val = weight * d_nee_contrib * light_contrib;
                        auto d_light_contrib = weight * d_nee_contrib * bsdf_val;
                        // d_phase_val is not used since it is a constant (we assume perfect importance sampling)
                        // auto d_phase_val = weight * sum(d_nee_contrib);

                        auto d_wi = Vector3{0, 0, 0};
                        auto d_wo = Vector3{0, 0, 0};
                        // bsdf_val = bsdf(material, shading_point, wi, wo, min_rough)
                        d_bsdf(material, surface_point, wi, wo, min_rough, d_bsdf_val,
                            d_material, d_shading_point, d_wi, d_wo);

                        auto d_ray_diff = RayDifferential{
                            Vector3{0, 0, 0}, Vector3{0, 0, 0},
                            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                        // light_contrib = eval_envmap(*scene.envmap, wo, ray_diff)
                        d_envmap_eval(*scene.envmap, wo, ray_diff, d_light_contrib,
                            *d_envmap, d_wo, d_ray_diff);

                        // wi = -incoming_ray.dir
                        d_incoming_ray.dir -= d_wi;
                    }
                }
            }
        }

        // BSDF or phase importance sampling
        const auto &directional_isect = directional_isects[pixel_id];
        if (directional_isect.valid()) {
            // We hit a surface
            const auto &light_shape = scene.shapes[directional_isect.shape_id];

            if (distance_squared(directional_point.position, p) > 1e-20f) {
                auto wo = directional_ray.dir;
                auto directional_val = Vector3{1, 1, 1};
                auto directional_pdf = Real(1);
                if (eval_phase) {
                    // Inside a medium
                    auto phase_function =
                        get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                    directional_pdf = phase_function_pdf(phase_function, wi, wo);
                    auto phase_eval = phase_function_eval(phase_function, wi, wo);
                    // Perfect importance sampling
                    directional_val = Vector3{phase_eval, phase_eval, phase_eval};
                } else {
                    // On a surface
                    const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                    if (surface_shape.material_id >= 0) {
                        const auto &material = scene.materials[surface_shape.material_id];
                        directional_pdf = bsdf_pdf(material, surface_point, wi, wo, min_rough);
                        directional_val = bsdf(material, surface_point, wi, wo, min_rough);
                    }
                }
                if (directional_pdf > 1e-20f) {
                    // next_throughput = path_transmittance * throughput * (directional_val / directional_pdf)
                    if (d_path_transmittances != nullptr) {
                        d_path_transmittances[pixel_id] +=
                            d_next_throughput * throughput * (directional_val / directional_pdf);
                    }
                    d_throughput += d_next_throughput * path_transmittance * (directional_val / directional_pdf);
                    auto d_directional_val = d_next_throughput * path_transmittance / directional_pdf;

                    if (light_shape.light_id >= 0) {
                        const auto &light = scene.area_lights[light_shape.light_id];
                        if (light.two_sided || dot(-wo, directional_point.shading_frame.n) > 0) {
                            auto light_contrib = light.intensity;
                            auto light_pmf = scene.light_pmf[light_shape.light_id];
                            auto light_area = scene.light_areas[light_shape.light_id];
                            auto inv_area = 1 / light_area;
                            auto geometry_term = fabs(dot(wo, directional_point.geom_normal)) /
                                distance_squared(directional_point.position, p);
                            auto pdf_nee = (light_pmf * inv_area) / geometry_term;
                            auto mis_weight = Real(1 / (1 + square((double)pdf_nee / (double)directional_pdf)));
                            auto scatter_contrib = (mis_weight / directional_pdf) *
                                directional_val * light_contrib;

                            // path_contrib = path_transmittance * throughput * (nee_contrib + scatter_contrib)
                            auto d_scatter_contrib = d_path_contrib * throughput * path_transmittance;
                            d_throughput += d_path_contrib * scatter_contrib * path_transmittance;
                            if (d_path_transmittances != nullptr) {
                                d_path_transmittances[pixel_id] += d_path_contrib * throughput * scatter_contrib;
                            }

                            // scatter_contrib = (mis_weight / directional_pdf) *
                            //     directional_val * light_contrib
                            auto weight = mis_weight / directional_pdf;
                            d_directional_val += weight * d_scatter_contrib * directional_val *
                                light_contrib;
                            auto d_light_contrib = weight * d_scatter_contrib *
                                directional_val;

                            // light_contrib = light.intensity
                            atomic_add(d_area_lights[light_shape.light_id].intensity, d_light_contrib);
                        }
                    }
                    auto d_wi = Vector3{0, 0, 0};
                    auto d_wo = d_next_ray.dir; // propagated from the next intersection
                    if (eval_phase) {
                        // We assume directional_val / directional_pdf == constant
                        // due to perfect importance sampling of phase function.
                        // So we do nothing here.
                    } else {
                        const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                        if (surface_shape.material_id >= 0) {
                            const auto &material = scene.materials[surface_shape.material_id];
                            // directional_val = bsdf(material, surface_point, wi, wo, min_rough)
                            d_bsdf(material, surface_point, wi, wo, min_rough, d_directional_val,
                                   d_material, d_shading_point, d_wi, d_wo);
                            // XXX: Ignore derivative w.r.t. pdf_bsdf since it causes high variance
                            // when propagating back from many bounces
                            // This is still correct since
                            // E[(\nabla f) / p] = \int (\nabla f) / p * p = \int (\nabla f)
                            // An intuitive way to think about this is that we are dividing the pdfs
                            // and multiplying MIS weights also for our gradient estimator
                        }
                    }
                    d_next_ray.dir += d_wo;
                    // wi = -incoming_ray.dir
                    d_incoming_ray.dir -= d_wi;
                }
            }
        } else if (scene.envmap != nullptr) {
            // Hit environment map
            auto wo = directional_ray.dir;
            auto directional_pdf = Real(1);
            auto directional_val = Vector3{1, 1, 1};
            if (eval_phase) {
                // Intersection with medium was made. Get PDF of phase function
                directional_pdf = phase_function_pdf(
                    get_phase_function(scene.mediums[medium_ids[pixel_id]]),
                    wo, wi);
                directional_val = Vector3{directional_pdf, directional_pdf, directional_pdf};
            } else {
                // Intersection with surface was made. Get BSDF PDF
                const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                const auto &material = scene.materials[surface_shape.material_id];
                directional_pdf = bsdf_pdf(material, surface_point, wi, wo, min_rough);

                if (directional_pdf > 1e-20f) {
                    directional_val = bsdf(material, surface_point, wi, wo, min_rough);
                }
            }

            // wo can be zero if sampling of the BSDF or phase function fails
            if (length_squared(wo) > 0 && directional_pdf > 1e-20f) {
                auto ray_diff = RayDifferential{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                    Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                auto light_contrib = envmap_eval(*scene.envmap, wo, ray_diff);
                auto envmap_id = scene.num_lights - 1;
                auto light_pmf = scene.light_pmf[envmap_id];
                auto pdf_nee = envmap_pdf(*scene.envmap, wo) * light_pmf;
                auto mis_weight = Real(1 / (1 + square((double)pdf_nee / (double)directional_pdf)));
                auto scatter_contrib = (mis_weight / directional_pdf) *
                    directional_val * light_contrib;

                // next_throughput = path_transmittance * throughput * (directional_val / directional_pdf)
                if (d_path_transmittances != nullptr) {
                    d_path_transmittances[pixel_id] +=
                        d_next_throughput * throughput * (directional_val / directional_pdf);
                }
                d_throughput += d_next_throughput * path_transmittance * (directional_val / directional_pdf);
                auto d_directional_val = d_next_throughput * path_transmittance / directional_pdf;

                // path_contrib = path_transmittance * throughput * (nee_contrib + scatter_contrib)
                auto d_scatter_contrib = d_path_contrib * throughput * path_transmittance;
                d_throughput += d_path_contrib * scatter_contrib * path_transmittance;
                if (d_path_transmittances != nullptr) {
                    d_path_transmittances[pixel_id] += d_path_contrib * throughput * scatter_contrib;
                }

                // scatter_contrib = (mis_weight / directional_pdf) *
                //     directional_val * light_contrib
                auto weight = mis_weight / directional_pdf;
                d_directional_val += weight * d_scatter_contrib * directional_val *
                    light_contrib;
                auto d_light_contrib = weight * d_scatter_contrib *
                    directional_val;

                auto d_wo = Vector3{0, 0, 0};
                auto d_ray_diff = RayDifferential{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                    Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                // light_contrib = eval_envmap(*scene.envmap, wo, ray_diff)
                d_envmap_eval(*scene.envmap, wo, ray_diff, d_light_contrib,
                              *d_envmap, d_wo, d_ray_diff);
                auto d_wi = Vector3{0, 0, 0};
                // bsdf_val = bsdf(material, shading_point, wi, wo)

                if (eval_phase) {
                    // We assume directional_val / directional_pdf == constant
                    // due to perfect importance sampling of phase function.
                    // So we do nothing here.
                } else {
                    // Do not compute d_bsdf if we had a medium interaction
                    d_bsdf(material, surface_point, wi, wo, min_rough, d_directional_val,
                           d_material, d_shading_point, d_wi, d_wo);
                    // XXX: Ignore derivative w.r.t. pdf_bsdf since it causes high variance
                    // when propagating back from many bounces
                    // This is still correct since
                    // E[(\nabla f) / p] = \int (\nabla f) / p * p = \int (\nabla f)
                    // An intuitive way to think about this is that we are dividing the pdfs
                    // and multiplying MIS weights also for our gradient estimator
                }

                // wi = -incoming_ray.dir
                d_incoming_ray.dir -= d_wi;
            }
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Vector3 *throughputs;
    const Vector3 *path_transmittances;
    const Ray *incoming_rays;
    const LightSample *light_samples;
    const DirectionalSample *directional_samples;
    const Intersection *surface_isects;
    const SurfacePoint *surface_points;
    const int *medium_ids;
    const Real *medium_distances;
    const Intersection *light_isects;
    const SurfacePoint *light_points;
    const Ray *light_rays;
    const Intersection *directional_isects;
    const SurfacePoint *directional_points;
    const Ray *directional_rays;
    const Real *min_roughness;
    const Real weight;
    const ChannelInfo channel_info;
    const float *d_rendered_image;
    const Vector3 *d_next_throughputs;
    DRay *d_next_rays;
    const SurfacePoint *d_next_points;
    DShape *d_shapes;
    DMaterial *d_materials;
    DAreaLight *d_area_lights;
    DEnvironmentMap *d_envmap;
    DMedium *d_mediums;
    Vector3 *d_throughputs;
    Vector3 *d_path_transmittances;
    DRay *d_incoming_rays;
    SurfacePoint *d_shading_points;
};

void accumulate_path_contribs(const Scene &scene,
                              const BufferView<int> &active_pixels,
                              const BufferView<Vector3> &throughputs,
                              const BufferView<Vector3> &path_transmittances,
                              // Information of the shading point
                              const BufferView<Ray> &incoming_rays,
                              const BufferView<Intersection> &surface_isects,
                              const BufferView<SurfacePoint> &surface_points,
                              const BufferView<int> &medium_ids,
                              const BufferView<Real> &medium_distances,
                              // Information of next event estimation
                              const BufferView<Intersection> &light_isects,
                              const BufferView<SurfacePoint> &light_points,
                              const BufferView<Ray> &light_rays,
                              const BufferView<Intersection> &light_int_isects,
                              const BufferView<SurfacePoint> &light_int_points,
                              const BufferView<Ray> &light_int_rays,
                              const BufferView<int> &light_medium_ids,
                              // Information of scattering
                              const BufferView<Intersection> &scatter_isects,
                              const BufferView<SurfacePoint> &scatter_points,
                              const BufferView<Ray> &scatter_rays,
                              const BufferView<Intersection> &next_isects,
                              const BufferView<SurfacePoint> &next_points,
                              const BufferView<Ray> &scatter_int_rays,
                              const BufferView<int> &scatter_medium_ids,
                              // Misc
                              const BufferView<Real> &min_roughness,
                              const Real weight,
                              const ChannelInfo &channel_info,
                              // Output
                              BufferView<Vector3> next_throughputs,
                              float *rendered_image,
                              BufferView<Real> edge_contribs) {
    parallel_for(path_contribs_accumulator{
        get_flatten_scene(scene),
        active_pixels.begin(),
        throughputs.begin(),
        path_transmittances.begin(),
        incoming_rays.begin(),
        surface_isects.begin(),
        surface_points.begin(),
        medium_ids.begin(),
        medium_distances.begin(),
        light_isects.begin(),
        light_points.begin(),
        light_rays.begin(),
        light_int_isects.begin(),
        light_int_points.begin(),
        light_int_rays.begin(),
        light_medium_ids.begin(),
        scatter_isects.begin(),
        scatter_points.begin(),
        scatter_rays.begin(),
        next_isects.begin(),
        next_points.begin(),
        scatter_int_rays.begin(),
        scatter_medium_ids.begin(),
        min_roughness.begin(),
        weight,
        channel_info,
        next_throughputs.begin(),
        rendered_image,
        edge_contribs.begin()}, active_pixels.size(), scene.use_gpu);
}

void d_accumulate_path_contribs(const Scene &scene,
                                const BufferView<int> &active_pixels,
                                const BufferView<Vector3> &throughputs,
                                const BufferView<Vector3> &path_transmittances,
                                const BufferView<Ray> &incoming_rays,
                                const BufferView<LightSample> &light_samples,
                                const BufferView<DirectionalSample> &directional_samples,
                                const BufferView<Intersection> &surface_isects,
                                const BufferView<SurfacePoint> &surface_points,
                                const BufferView<int> &medium_ids,
                                const BufferView<Real> &medium_distances,
                                const BufferView<Intersection> &light_isects,
                                const BufferView<SurfacePoint> &light_points,
                                const BufferView<Ray> &light_rays,
                                const BufferView<Intersection> &directional_isects,
                                const BufferView<SurfacePoint> &directional_points,
                                const BufferView<Ray> &directional_rays,
                                const BufferView<Real> &min_roughness,
                                const Real weight,
                                const ChannelInfo &channel_info,
                                const float *d_rendered_image,
                                const BufferView<Vector3> &d_next_throughputs,
                                BufferView<DRay> d_next_rays,
                                const BufferView<SurfacePoint> &d_next_points,
                                DScene *d_scene,
                                BufferView<Vector3> d_throughputs,
                                BufferView<Vector3> d_path_transmittances,
                                BufferView<DRay> d_incoming_rays,
                                BufferView<SurfacePoint> d_shading_points) {
    parallel_for(d_path_contribs_accumulator{
        get_flatten_scene(scene),
        active_pixels.begin(),
        throughputs.begin(),
        path_transmittances.begin(),
        incoming_rays.begin(),
        light_samples.begin(),
        directional_samples.begin(),
        surface_isects.begin(),
        surface_points.begin(),
        medium_ids.begin(),
        medium_distances.begin(),
        light_isects.begin(),
        light_points.begin(),
        light_rays.begin(),
        directional_isects.begin(),
        directional_points.begin(),
        directional_rays.begin(),
        min_roughness.begin(),
        weight,
        channel_info,
        d_rendered_image,
        d_next_throughputs.begin(),
        d_next_rays.begin(),
        d_next_points.begin(),
        d_scene->shapes.data,
        d_scene->materials.data,
        d_scene->area_lights.data,
        d_scene->envmap,
        d_scene->mediums.data,
        d_throughputs.begin(),
        d_path_transmittances.begin(),
        d_incoming_rays.begin(),
        d_shading_points.begin()},
        active_pixels.size(), scene.use_gpu);
}
