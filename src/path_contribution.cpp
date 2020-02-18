#include "path_contribution.h"
#include "medium.h"
#include "parallel.h"
#include "scene.h"

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
        const auto &bsdf_isect = bsdf_isects[pixel_id];
        const auto &bsdf_point = bsdf_points[pixel_id];
        const auto &bsdf_ray = bsdf_rays[pixel_id];
        const auto &min_rough = min_roughness[pixel_id];
        auto &next_throughput = next_throughputs[pixel_id];

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
                            // Compute phase function instead of the bsdf if we are inside medium.
                            // We assume we are perfectly importance sampling the phase functions,
                            // and the phase functions integrates to 1. So we don't need "phase_val".
                            // Still we need the phase function PDF for MIS.
                            auto pdf_phase = phase_function_pdf(
                                get_phase_function(scene.mediums[medium_ids[pixel_id]]),
                                wo, wi);
                            auto mis_weight = Real(1 / (1 + square((double)pdf_phase / (double)pdf_nee)));
                            nee_contrib =
                                (mis_weight * geometry_term / pdf_nee) * light_contrib;
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
                        if (nee_transmittances != nullptr) {
                            nee_contrib *= nee_transmittances[pixel_id];
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
                        // Compute phase function instead of the bsdf if we are inside medium.
                        // We assume we are perfectly importance sampling the phase functions,
                        // and the phase functions integrates to 1. So we don't need "phase_val".
                        // Still we need the phase function PDF for MIS.
                        auto pdf_phase = phase_function_pdf(
                            get_phase_function(scene.mediums[medium_ids[pixel_id]]),
                            wo, wi);
                        auto mis_weight = Real(1 / (1 + square((double)pdf_phase / (double)pdf_nee)));
                        nee_contrib = (mis_weight / pdf_nee) * light_contrib;
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
                    if (nee_transmittances != nullptr) {
                        nee_contrib *= nee_transmittances[pixel_id];
                    }
                }
            }
        }

        // BSDF or phase function importance sampling.
        // Also update throughput, transmittance is updated in another kernel.
        auto scatter_contrib = Vector3{0, 0, 0};
        if (bsdf_isect.valid()) { // We hit a surface
            const auto &bsdf_shape = scene.shapes[bsdf_isect.shape_id];
            auto dir = bsdf_point.position - p;
            auto dist_sq = length_squared(dir);
            if (dist_sq > 1e-20f) {
                auto wo = dir / sqrt(dist_sq);
                auto directional_val = Vector3{1, 1, 1};
                auto directional_pdf = Real(1);
                if (eval_phase) {
                    // Inside a medium
                    directional_pdf = phase_function_pdf(
                        get_phase_function(scene.mediums[medium_ids[pixel_id]]),
                        wo, wi);
                    // Perfect importance sampling
                    directional_val = Vector3{directional_pdf, directional_pdf, directional_pdf};
                } else {
                    // On a surface
                    const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                    if (surface_shape.material_id >= 0) {
                        const auto &material = scene.materials[surface_shape.material_id];
                        directional_pdf = bsdf_pdf(material, surface_point, wi, wo, min_rough);
                        if (directional_pdf > 1e-20f) {
                            directional_val = bsdf(material, surface_point, wi, wo, min_rough);
                        }
                    }
                }
                if (directional_pdf > 1e-20f) {
                    if (bsdf_shape.light_id >= 0) {
                        const auto &light = scene.area_lights[bsdf_shape.light_id];
                        if (light.two_sided || dot(-wo, bsdf_point.shading_frame.n) > 0) {
                            auto light_contrib = light.intensity;
                            auto light_pmf = scene.light_pmf[bsdf_shape.light_id];
                            auto light_area = scene.light_areas[bsdf_shape.light_id];
                            auto inv_area = 1 / light_area;
                            auto geometry_term = fabs(dot(wo, bsdf_point.geom_normal)) / dist_sq;
                            auto pdf_nee = (light_pmf * inv_area) / geometry_term;
                            auto mis_weight = Real(1 / (1 + square((double)pdf_nee / (double)directional_pdf)));
                            scatter_contrib = (mis_weight / directional_pdf) * directional_val * light_contrib;
                        }
                    }
                    // Update throughput
                    next_throughput = throughput * (directional_val / directional_pdf);
                    if (transmittances != nullptr) {
                        next_throughput *= transmittances[pixel_id];
                        scatter_contrib *= transmittances[pixel_id];
                    }
                }
            }
        } else {
            auto wo = bsdf_ray.dir;
            // wo can be zero when bsdf_sample failed
            if (length_squared(wo) > 0) {
                auto directional_val = Vector3{1, 1, 1};
                auto directional_pdf = Real(1);
                if (eval_phase) {
                    // Inside a medium
                    directional_pdf = phase_function_pdf(
                        get_phase_function(scene.mediums[medium_ids[pixel_id]]),
                        wo, wi);
                    // Perfect importance sampling
                    directional_val = Vector3{directional_pdf, directional_pdf, directional_pdf};
                } else {
                    // On a surface
                    const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                    if (surface_shape.material_id >= 0) {
                        const auto &material = scene.materials[surface_shape.material_id];
                        directional_pdf = bsdf_pdf(material, surface_point, wi, wo, min_rough);
                        if (directional_pdf > 1e-20f) {
                            directional_val = bsdf(material, surface_point, wi, wo, min_rough);
                        }
                    }
                }
                if (scene.envmap != nullptr && directional_pdf > 1e-20f) {
                    // Hit an environment map
                    // XXX: For now we don't use ray differentials for envmap
                    //      A proper approach might be to use a filter radius based on sampling density?
                    RayDifferential ray_diff{Vector3{0, 0, 0}, Vector3{0, 0, 0},
                                             Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                    auto light_contrib = envmap_eval(*scene.envmap, wo, ray_diff);
                    auto envmap_id = scene.num_lights - 1;
                    auto light_pmf = scene.light_pmf[envmap_id];
                    auto pdf_nee = envmap_pdf(*scene.envmap, wo) * light_pmf;
                    auto mis_weight = Real(1 / (1 + square((double)pdf_nee / (double)directional_pdf)));
                    scatter_contrib = (mis_weight / directional_pdf) * directional_val * light_contrib;
                }
                next_throughput = throughput * (directional_val / directional_pdf);
                if (transmittances != nullptr) {
                    next_throughput *= transmittances[pixel_id];
                    scatter_contrib *= transmittances[pixel_id];
                }
            }
        }

        auto path_contrib = throughput * (nee_contrib + scatter_contrib);
        assert(isfinite(nee_contrib));
        assert(isfinite(scatter_contrib));
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
    const Vector3 *transmittances;
    const Vector3 *nee_transmittances;
    const Ray *incoming_rays;
    const Intersection *surface_isects;
    const SurfacePoint *surface_points;
    const int *medium_ids;
    const Real *medium_distances;
    const Intersection *light_isects;
    const SurfacePoint *light_points;
    const Ray *light_rays;
    const Intersection *bsdf_isects;
    const SurfacePoint *bsdf_points;
    const Ray *bsdf_rays;
    const Real *min_roughness;
    const Real weight;
    const ChannelInfo channel_info;
    Vector3 *next_throughputs;
    float *rendered_image;
    Real *edge_contribs;
};

struct d_path_contribs_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &throughput = throughputs[pixel_id];
        const auto &nee_transmittance = nee_transmittances[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        // const auto &incoming_ray_differential = incoming_ray_differentials[pixel_id];
        const auto &bsdf_ray_differential = bsdf_ray_differentials[pixel_id];
        const auto &surface_isect = surface_isects[pixel_id];
        const auto &surface_point = surface_points[pixel_id];
        const auto &light_isect = light_isects[pixel_id];
        const auto &light_ray = light_rays[pixel_id];
        const auto &min_rough = min_roughness[pixel_id];

        auto &d_throughput = d_throughputs[pixel_id];
        auto &d_incoming_ray = d_incoming_rays[pixel_id];
        auto &d_incoming_ray_differential = d_incoming_ray_differentials[pixel_id];
        auto &d_shading_point = d_shading_points[pixel_id];

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
        d_incoming_ray_differential = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        d_shading_point = SurfacePoint::zero();

        // Next event estimation
        auto nee_contrib = Vector3{0, 0, 0};
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
                            auto pdf_phase = phase_function_pdf(
                                get_phase_function(scene.mediums[medium_ids[pixel_id]]), wo, wi);
                            auto mis_weight = Real(1 / (1 + square((double)pdf_phase / (double)pdf_nee)));
                            nee_contrib =
                                (mis_weight * geometry_term / pdf_nee) * light_contrib * nee_transmittance;

                            // path_contrib = throughput * (nee_contrib + scatter_contrib)
                            auto d_nee_contrib = d_path_contrib * throughput;
                            d_throughput += d_path_contrib * nee_contrib;

                            auto weight = mis_weight / pdf_nee;
                            // nee_contrib = (weight * geometry_term) *
                            //                nee_transmittance * light_contrib
                            // Ignore derivatives of MIS & PMF
                            auto d_geometry_term =
                                weight * sum(d_nee_contrib * nee_transmittance * light_contrib);
                            auto d_nee_transmittances =
                                weight * d_nee_contrib * geometry_term * light_contrib;
                            auto d_light_contrib =
                                weight * d_nee_contrib * geometry_term * nee_transmittance;
                            auto d_weight = geometry_term *
                                sum(d_nee_contrib * nee_transmittance * light_contrib);

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
                            auto d_wo = d_cos_light * light_point.geom_normal;
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
                            // TODO Figure out how to propagate this properly
                            //d_incoming_ray.dir -= d_ray.dir;

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

                            nee_contrib = (mis_weight * geometry_term / pdf_nee) *
                                            bsdf_val * light_contrib;

                            // path_contrib = throughput * (nee_contrib + scatter_contrib)
                            auto d_nee_contrib = d_path_contrib * throughput;
                            d_throughput += d_path_contrib * nee_contrib;

                            auto weight = mis_weight / pdf_nee;
                            // nee_contrib = (weight * geometry_term) *
                            //                bsdf_val * light_contrib
                            // Ignore derivatives of MIS weight & PMF
                            auto d_weight = geometry_term *
                                sum(d_nee_contrib * bsdf_val * light_contrib);
                            // weight = mis_weight / pdf_nee
                            auto d_pdf_nee = -d_weight * weight / pdf_nee;
                            // nee_contrib = (weight * geometry_term) *
                            //                bsdf_val * light_contrib
                            auto d_geometry_term = weight * sum(d_nee_contrib * bsdf_val * light_contrib);
                            auto d_bsdf_val = weight * d_nee_contrib * geometry_term * light_contrib;
                            auto d_light_contrib = weight * d_nee_contrib * geometry_term * bsdf_val;
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
                            auto d_wo = d_cos_light * light_point.geom_normal;
                            auto d_light_point = SurfacePoint::zero();
                            d_light_point.geom_normal = d_cos_light * wo;
                            // bsdf_val = bsdf(material, surface_point, wi, wo, min_rough)
                            auto d_wi = Vector3{0, 0, 0};
                            d_bsdf(material, surface_point, wi, wo, min_rough, d_bsdf_val,
                                d_material, d_shading_point, d_wi, d_wo);
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
                        // Compute the phase function PDF instead of the BSDF
                        // if we are inside of a medium.
                        auto pdf_phase = phase_function_pdf(
                            get_phase_function(scene.mediums[medium_ids[pixel_id]]),
                            wo, wi);
                        auto mis_weight = Real(1 / (1 + square((double)pdf_phase / (double)pdf_nee)));
                        nee_contrib = (mis_weight / pdf_nee) * light_contrib * nee_transmittance;

                        // path_contrib = throughput * (nee_contrib + scatter contrib)
                        auto d_nee_contrib = d_path_contrib * throughput;
                        d_throughput += d_path_contrib * nee_contrib;

                        auto weight = mis_weight / pdf_nee;

                        // nee_contrib = weight * light_contrib * nee_transmittances
                        auto d_light_contrib = weight * d_nee_contrib * nee_transmittance;
                        auto d_wo = Vector3{0, 0, 0};
                        auto d_ray_diff = RayDifferential{
                            Vector3{0, 0, 0}, Vector3{0, 0, 0},
                            Vector3{0, 0, 0}, Vector3{0, 0, 0}};

                        // light_contrib = eval_envmap(*scene.envmap, wo, ray_diff)
                        d_envmap_eval(*scene.envmap, wo, ray_diff, d_light_contrib,
                            *d_envmap, d_wo, d_ray_diff);

                        // See comment above about why we do not find the gradient
                        // of phase_function_pdf()
                        // auto d_pdf_phase = d_phase_function(
                        //     get_phase_function(medium),
                        //     wi, wo, d_wi, d_wo);
                        // auto d_ray = DRay{Vector3{0, 0, 0}, Vector3{0, 0, 0}};

                        // wi = -incoming_ray.dir;
                        // TODO Uncomment this later
                        //d_incoming_ray.dir -= d_ray.dir;
                    } else {
                        auto bsdf_val = bsdf(material, surface_point, wi, wo, min_rough);
                        auto pdf_bsdf = bsdf_pdf(material, surface_point, wi, wo, min_rough);
                        auto mis_weight = Real(1 / (1 + square((double)pdf_bsdf / (double)pdf_nee)));
                        auto nee_contrib = (mis_weight / pdf_nee) * bsdf_val * light_contrib;

                        // path_contrib = throughput * (nee_contrib + scatter_contrib)
                        auto d_nee_contrib = d_path_contrib * throughput;
                        d_throughput += d_path_contrib * nee_contrib;

                        auto weight = mis_weight / pdf_nee;
                        // nee_contrib = weight * bsdf_val * light_contrib
                        // Ignore derivatives of MIS weight & pdf
                        auto d_bsdf_val = weight * d_nee_contrib * light_contrib;
                        auto d_light_contrib = weight * d_nee_contrib * bsdf_val;
                        auto d_wo = Vector3{0, 0, 0};
                        auto d_ray_diff = RayDifferential{
                            Vector3{0, 0, 0}, Vector3{0, 0, 0},
                            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                        // light_contrib = eval_envmap(*scene.envmap, wo, ray_diff)
                        d_envmap_eval(*scene.envmap, wo, ray_diff, d_light_contrib,
                            *d_envmap, d_wo, d_ray_diff);
                        // bsdf_val = bsdf(material, shading_point, wi, wo, min_rough)
                        auto d_wi = Vector3{0, 0, 0};
                        d_bsdf(material, surface_point, wi, wo, min_rough, d_bsdf_val,
                            d_material, d_shading_point, d_wi, d_wo);
                        // wi = -incoming_ray.dir
                        d_incoming_ray.dir -= d_wi;
                    }
                }
            }
        }

        // BSDF or phase importance sampling
        const auto &bsdf_isect = bsdf_isects[pixel_id];
        auto scatter_contrib = Vector3{0, 0, 0};
        if (bsdf_isect.valid()) {
            // We hit a surface
            const auto &bsdf_shape = scene.shapes[bsdf_isect.shape_id];
            // const auto &bsdf_sample = bsdf_samples[pixel_id];
            const auto &bsdf_point = bsdf_points[pixel_id];
            const auto &d_next_ray = d_next_rays[pixel_id];
            const auto &d_next_ray_differential = d_next_ray_differentials[pixel_id];
            const auto &d_next_point = d_next_points[pixel_id];
            const auto &d_next_throughput = d_next_throughputs[pixel_id];

            // Initialize bsdf vertex derivatives
            auto dir = bsdf_point.position - p;
            auto dist_sq = length_squared(dir);
            auto wo = dir / sqrt(dist_sq);
            auto directional_val = Vector3{1, 1, 1};
            auto directional_pdf = Real(1);
            if (eval_phase) {
                // Inside a medium
                directional_pdf = phase_function_pdf(
                    get_phase_function(scene.mediums[medium_ids[pixel_id]]),
                    wo, wi);
                // Perfect importance sampling
                directional_val = Vector3{directional_pdf, directional_pdf, directional_pdf};
            } else {
                // On a surface
                const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                const auto &material = scene.materials[surface_shape.material_id];
                directional_pdf = bsdf_pdf(material, surface_point, wi, wo, min_rough);
                if (directional_pdf > 1e-20f) {
                    directional_val = bsdf(material, surface_point, wi, wo, min_rough);
                }
            }
            if (directional_pdf > 1e-20f) {
                if (bsdf_shape.light_id >= 0) {
                    const auto &light = scene.area_lights[bsdf_shape.light_id];
                    if (light.two_sided || dot(-wo, bsdf_point.shading_frame.n) > 0) {
                        auto light_contrib = light.intensity;
                        auto light_pmf = scene.light_pmf[bsdf_shape.light_id];
                        auto light_area = scene.light_areas[bsdf_shape.light_id];
                        auto inv_area = 1 / light_area;
                        auto geometry_term = fabs(dot(wo, bsdf_point.geom_normal)) / dist_sq;
                        auto pdf_nee = (light_pmf * inv_area) / geometry_term;
                        auto mis_weight = Real(1 / (1 + square((double)pdf_nee / (double)directional_pdf)));
                        scatter_contrib = (mis_weight / directional_pdf) * directional_val * light_contrib;
                    }
                }
            }

            if (directional_pdf > 1e-20f) {
                Vector3 d_bsdf_v_p[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                Vector3 d_bsdf_v_n[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                Vector2 d_bsdf_v_uv[3] = {Vector2{0, 0}, Vector2{0, 0}};
                Vector3 d_bsdf_v_c[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};

                auto scatter_bsdf = directional_val / directional_pdf;

                // next_throughput = throughput * scatter_bsdf
                d_throughput += d_next_throughput * scatter_bsdf;
                auto d_scatter_bsdf = d_next_throughput * throughput;
                // scatter_bsdf = bsdf_val / pdf_bsdf
                auto d_bsdf_val = d_scatter_bsdf / directional_pdf;
                // XXX: Ignore derivative w.r.t. pdf_bsdf since it causes high variance
                // when propagating back from many bounces
                // This is still correct since
                // E[(\nabla f) / p] = \int (\nabla f) / p * p = \int (\nabla f)
                // An intuitive way to think about this is that we are dividing the pdfs
                // and multiplying MIS weights also for our gradient estimator

                // auto d_pdf_bsdf = -sum(d_scatter_bsdf * scatter_bsdf) / pdf_bsdf;

                if (bsdf_shape.light_id >= 0) {
                    const auto &light = scene.area_lights[bsdf_shape.light_id];
                    if (light.two_sided || dot(-wo, bsdf_point.shading_frame.n) > 0) {
                        auto geometry_term = fabs(dot(wo, bsdf_point.geom_normal)) / dist_sq;
                        const auto &light = scene.area_lights[bsdf_shape.light_id];
                        auto light_contrib = light.intensity;
                        auto light_pmf = scene.light_pmf[bsdf_shape.light_id];
                        auto light_area = scene.light_areas[bsdf_shape.light_id];
                        auto inv_area = 1 / light_area;
                        auto pdf_nee = (light_pmf * inv_area) / geometry_term;
                        auto mis_weight = Real(1 / (1 + square((double)pdf_nee / (double)directional_pdf)));
                        auto scatter_contrib = (mis_weight / directional_pdf) * directional_val * light_contrib;

                        // path_contrib = throughput * (nee_contrib + scatter_contrib)
                        auto d_scatter_contrib = d_path_contrib * throughput;
                        d_throughput += d_path_contrib * scatter_contrib;
                        auto weight = mis_weight / directional_pdf;
                        // scatter_contrib = weight * bsdf_val * light_contrib

                        // auto d_weight = sum(d_scatter_contrib * bsdf_val * light_contrib);
                        // Ignore derivatives of MIS weight & pdf_bsdf
                        // weight = mis_weight / pdf_bsdf
                        // d_pdf_bsdf += -d_weight * weight / pdf_bsdf;
                        d_bsdf_val += weight * d_scatter_contrib * light_contrib;
                        auto d_light_contrib = weight * d_scatter_contrib * directional_val;
                        // light_contrib = light.intensity
                        atomic_add(d_area_lights[bsdf_shape.light_id].intensity, d_light_contrib);
                    }
                }

                auto d_wi = Vector3{0, 0, 0};
                auto d_wo = d_next_ray.dir;
                // pdf_bsdf = bsdf_pdf(material, shading_point, wi, wo, min_rough)
                // d_bsdf_pdf(material, shading_point, wi, wo, min_rough, d_pdf_bsdf,
                //            d_roughness_tex, d_shading_point, d_wi, d_wo);
                // bsdf_val = bsdf(material, shading_point, wi, wo)
                if (!eval_phase) {
                    // Only compute d_bsdf if we did not have a medium interaction
                    d_bsdf(material, surface_point, wi, wo, min_rough, d_bsdf_val,
                        d_material, d_shading_point, d_wi, d_wo);
                }

                // wo = dir / sqrt(dist_sq)
                auto d_dir = d_wo / sqrt(dist_sq);
                auto d_sqrt_dist_sq = -sum(d_wo * dir) / dist_sq;
                auto d_dist_sq = 0.5f * d_sqrt_dist_sq / sqrt(dist_sq);
                // dist_sq = length_squared(dir)
                d_dir += d_length_squared(dir, d_dist_sq);

                auto d_bsdf_point = d_next_point;
                // dir = bsdf_point.position - p
                d_bsdf_point.position += d_dir;
                // d_shading_point.position -= d_dir; (see below)

                // bsdf intersection
                DRay d_ray;
                RayDifferential d_bsdf_ray_differential{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                    Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                d_intersect_shape(bsdf_shape,
                                  bsdf_isect.tri_id,
                                  Ray{surface_point.position, wo},
                                  bsdf_ray_differential,
                                  d_bsdf_point,
                                  d_next_ray_differential,
                                  d_ray,
                                  d_bsdf_ray_differential,
                                  d_bsdf_v_p,
                                  d_bsdf_v_n,
                                  d_bsdf_v_uv,
                                  d_bsdf_v_c);

                // XXX HACK: diffuse interreflection causes a lot of noise
                // on position derivatives but has small impact on the final derivatives,
                // so we ignore them here.
                // A properer way is to come up with an importance sampling distribution,
                // or use a lot more samples
                if (min_rough > 0.01f) {
                    d_shading_point.position -= d_dir;
                    d_shading_point.position += d_ray.org;
                }
                d_wo += d_ray.dir;

                // We ignore backpropagation to bsdf importance sampling
                // d_bsdf_sample(material,
                //               shading_point,
                //               wi,
                //               bsdf_sample,
                //               min_rough,
                //               incoming_ray_differential,
                //               d_wo,
                //               d_bsdf_ray_differential,
                //               d_roughness_tex,
                //               d_shading_point,
                //               d_wi,
                //               d_incoming_ray_differential);

                // wi = -incoming_ray.dir
                d_incoming_ray.dir -= d_wi;

                // Accumulate derivatives
                auto bsdf_tri_index = get_indices(bsdf_shape, bsdf_isect.tri_id);
                atomic_add(&d_shapes[bsdf_isect.shape_id].vertices[3 * bsdf_tri_index[0]],
                    d_bsdf_v_p[0]);
                atomic_add(&d_shapes[bsdf_isect.shape_id].vertices[3 * bsdf_tri_index[1]],
                    d_bsdf_v_p[1]);
                atomic_add(&d_shapes[bsdf_isect.shape_id].vertices[3 * bsdf_tri_index[2]],
                    d_bsdf_v_p[2]);
                if (has_uvs(bsdf_shape)) {
                    auto uv_tri_ind = bsdf_tri_index;
                    if (bsdf_shape.uv_indices != nullptr) {
                        uv_tri_ind = get_uv_indices(bsdf_shape, bsdf_isect.tri_id);
                    }
                    atomic_add(&d_shapes[bsdf_isect.shape_id].uvs[2 * uv_tri_ind[0]],
                        d_bsdf_v_uv[0]);
                    atomic_add(&d_shapes[bsdf_isect.shape_id].uvs[2 * uv_tri_ind[1]],
                        d_bsdf_v_uv[1]);
                    atomic_add(&d_shapes[bsdf_isect.shape_id].uvs[2 * uv_tri_ind[2]],
                        d_bsdf_v_uv[2]);
                }
                if (has_shading_normals(bsdf_shape)) {
                    auto normal_tri_ind = bsdf_tri_index;
                    if (bsdf_shape.normal_indices != nullptr) {
                        normal_tri_ind = get_normal_indices(bsdf_shape, bsdf_isect.tri_id);
                    }
                    atomic_add(&d_shapes[bsdf_isect.shape_id].normals[3 * normal_tri_ind[0]],
                        d_bsdf_v_n[0]);
                    atomic_add(&d_shapes[bsdf_isect.shape_id].normals[3 * normal_tri_ind[1]],
                        d_bsdf_v_n[1]);
                    atomic_add(&d_shapes[bsdf_isect.shape_id].normals[3 * normal_tri_ind[2]],
                        d_bsdf_v_n[2]);
                }
                if (has_colors(bsdf_shape)) {
                    atomic_add(&d_shapes[bsdf_isect.shape_id].colors[3 * bsdf_tri_index[0]],
                        d_bsdf_v_c[0]);
                    atomic_add(&d_shapes[bsdf_isect.shape_id].colors[3 * bsdf_tri_index[1]],
                        d_bsdf_v_c[1]);
                    atomic_add(&d_shapes[bsdf_isect.shape_id].colors[3 * bsdf_tri_index[2]],
                        d_bsdf_v_c[2]);
                }
            }
        } else if (scene.envmap != nullptr) {
            // Hit environment map
            const auto &bsdf_ray = bsdf_rays[pixel_id];

            auto wo = bsdf_ray.dir;
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
                auto scatter_contrib = (mis_weight / directional_pdf) * directional_val * light_contrib;

                // path_contrib = throughput * (nee_contrib + scatter_contrib)
                auto d_scatter_contrib = d_path_contrib * throughput;
                d_throughput += d_path_contrib * scatter_contrib;
                auto weight = mis_weight / directional_pdf;

                // scatter_contrib = weight * bsdf_val * light_contrib
                // auto d_weight = sum(d_scatter_contrib * bsdf_val * light_contrib);
                // Ignore derivatives of MIS weight and pdf
                // XXX: unlike the case of mesh light sources, we don't propagate to
                //      the sampling procedure, since it causes higher variance when
                //      there is a huge gradients in d_envmap_eval()
                // weight = mis_weight / pdf_bsdf
                // auto d_pdf_bsdf = -d_weight * weight / pdf_bsdf;
                auto d_bsdf_val = weight * d_scatter_contrib * light_contrib;
                auto d_light_contrib = weight * d_scatter_contrib * directional_pdf;

                auto d_wo = Vector3{0, 0, 0};
                auto d_ray_diff = RayDifferential{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                    Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                // light_contrib = eval_envmap(*scene.envmap, wo, ray_diff)
                d_envmap_eval(*scene.envmap, wo, ray_diff, d_light_contrib,
                              *d_envmap, d_wo, d_ray_diff);
                auto d_wi = Vector3{0, 0, 0};
                // bsdf_val = bsdf(material, shading_point, wi, wo)

                if (!eval_phase) {
                    // Do not compute d_bsdf if we had a medium interaction
                    d_bsdf(material, surface_point, wi, wo, min_rough, d_bsdf_val,
                        d_material, d_shading_point, d_wi, d_wo);
                }

                // pdf_bsdf = bsdf_pdf(material, shading_point, wi, wo, min_rough)
                // d_bsdf_pdf(material, shading_point, wi, wo, min_rough, d_pdf_bsdf,
                //            d_roughness_tex, d_shading_point, d_wi, d_wo);

                // sample bsdf direction
                // auto d_bsdf_ray_differential = RayDifferential{
                //     Vector3{0, 0, 0}, Vector3{0, 0, 0},
                //     Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                // d_bsdf_sample(material,
                //               shading_point,
                //               wi,
                //               bsdf_sample,
                //               min_rough,
                //               incoming_ray_differential,
                //               d_wo,
                //               d_bsdf_ray_differential,
                //               d_roughness_tex,
                //               d_shading_point,
                //               d_wi,
                //               d_incoming_ray_differential);

                // wi = -incoming_ray.dir
                d_incoming_ray.dir -= d_wi;
            }
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Vector3 *throughputs;
    const Vector3 *transmittances;
    const Vector3 *nee_transmittances;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const LightSample *light_samples;
    const DirectionalSample *directional_samples;
    const Intersection *surface_isects;
    const SurfacePoint *surface_points;
    const int *medium_ids;
    const Real *medium_distances;
    const Intersection *light_isects;
    const SurfacePoint *light_points;
    const Ray *light_rays;
    const Intersection *bsdf_isects;
    const SurfacePoint *bsdf_points;
    const Ray *bsdf_rays;
    const RayDifferential *bsdf_ray_differentials;
    const Real *min_roughness;
    const Real weight;
    const ChannelInfo channel_info;
    const float *d_rendered_image;
    const Vector3 *d_next_throughputs;
    const DRay *d_next_rays;
    const RayDifferential *d_next_ray_differentials;
    const SurfacePoint *d_next_points;
    DShape *d_shapes;
    DMaterial *d_materials;
    DAreaLight *d_area_lights;
    DEnvironmentMap *d_envmap;
    DMedium *d_mediums;
    Vector3 *d_throughputs;
    DRay *d_incoming_rays;
    RayDifferential *d_incoming_ray_differentials;
    SurfacePoint *d_shading_points;
};

void accumulate_path_contribs(const Scene &scene,
                              const BufferView<int> &active_pixels,
                              const BufferView<Vector3> &throughputs,
                              const BufferView<Vector3> &transmittances,
                              const BufferView<Vector3> &nee_transmittances,
                              const BufferView<Ray> &incoming_rays,
                              const BufferView<Intersection> &surface_isects,
                              const BufferView<SurfacePoint> &surface_points,
                              const BufferView<int> &medium_ids,
                              const BufferView<Real> &medium_distances,
                              const BufferView<Intersection> &light_isects,
                              const BufferView<SurfacePoint> &light_points,
                              const BufferView<Ray> &light_rays,
                              const BufferView<Intersection> &bsdf_isects,
                              const BufferView<SurfacePoint> &bsdf_points,
                              const BufferView<Ray> &bsdf_rays,
                              const BufferView<Real> &min_roughness,
                              const Real weight,
                              const ChannelInfo &channel_info,
                              BufferView<Vector3> next_throughputs,
                              float *rendered_image,
                              BufferView<Real> edge_contribs) {
    parallel_for(path_contribs_accumulator{
        get_flatten_scene(scene),
        active_pixels.begin(),
        throughputs.begin(),
        transmittances.begin(),
        nee_transmittances.begin(),
        incoming_rays.begin(),
        surface_isects.begin(),
        surface_points.begin(),
        medium_ids.begin(),
        medium_distances.begin(),
        light_isects.begin(),
        light_points.begin(),
        light_rays.begin(),
        bsdf_isects.begin(),
        bsdf_points.begin(),
        bsdf_rays.begin(),
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
                                const BufferView<Vector3> &transmittances,
                                const BufferView<Vector3> &nee_transmittances,
                                const BufferView<Ray> &incoming_rays,
                                const BufferView<RayDifferential> &ray_differentials,
                                const BufferView<LightSample> &light_samples,
                                const BufferView<DirectionalSample> &directional_samples,
                                const BufferView<Intersection> &surface_isects,
                                const BufferView<SurfacePoint> &surface_points,
                                const BufferView<int> &medium_ids,
                                const BufferView<Real> &medium_distances,
                                const BufferView<Intersection> &light_isects,
                                const BufferView<SurfacePoint> &light_points,
                                const BufferView<Ray> &light_rays,
                                const BufferView<Intersection> &bsdf_isects,
                                const BufferView<SurfacePoint> &bsdf_points,
                                const BufferView<Ray> &bsdf_rays,
                                const BufferView<RayDifferential> &bsdf_ray_differentials,
                                const BufferView<Real> &min_roughness,
                                const Real weight,
                                const ChannelInfo &channel_info,
                                const float *d_rendered_image,
                                const BufferView<Vector3> &d_next_throughputs,
                                const BufferView<DRay> &d_next_rays,
                                const BufferView<RayDifferential> &d_next_ray_differentials,
                                const BufferView<SurfacePoint> &d_next_points,
                                DScene *d_scene,
                                BufferView<Vector3> d_throughputs,
                                BufferView<DRay> d_incoming_rays,
                                BufferView<RayDifferential> d_incoming_ray_differentials,
                                BufferView<SurfacePoint> d_shading_points) {
    parallel_for(d_path_contribs_accumulator{
        get_flatten_scene(scene),
        active_pixels.begin(),
        throughputs.begin(),
        transmittances.begin(),
        nee_transmittances.begin(),
        incoming_rays.begin(),
        ray_differentials.begin(),
        light_samples.begin(),
        directional_samples.begin(),
        surface_isects.begin(),
        surface_points.begin(),
        medium_ids.begin(),
        medium_distances.begin(),
        light_isects.begin(),
        light_points.begin(),
        light_rays.begin(),
        bsdf_isects.begin(),
        bsdf_points.begin(),
        bsdf_rays.begin(),
        bsdf_ray_differentials.begin(),
        min_roughness.begin(),
        weight,
        channel_info,
        d_rendered_image,
        d_next_throughputs.begin(),
        d_next_rays.begin(),
        d_next_ray_differentials.begin(),
        d_next_points.begin(),
        d_scene->shapes.data,
        d_scene->materials.data,
        d_scene->area_lights.data,
        d_scene->envmap,
        d_scene->mediums.data,
        d_throughputs.begin(),
        d_incoming_rays.begin(),
        d_incoming_ray_differentials.begin(),
        d_shading_points.begin()},
        active_pixels.size(), scene.use_gpu);
}