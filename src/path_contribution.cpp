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
    if (int_isect.valid()) {
        const auto &shape = scene.shapes[int_isect.shape_id];
        if (int_isect == tgt_isect) {
            // There is no intermediate intersect
            if (medium_id != -1) {
                const auto &medium = scene.mediums[medium_id];
                auto r = Ray{org_ray.org, org_ray.dir, Real(0),
                    distance(org_ray.org, tgt_point.position)};
                return transmittance(medium, r);
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
                        auto r = Ray{int_point.position, org_ray.dir, Real(0),
                                     distance(org_ray.org, int_point.position)};
                        tr0 = transmittance(medium, r);
                    }
                    if (int_medium_id != -1) {
                        auto r = Ray{};
                        if (!tgt_isect.valid()) {
                            r = Ray{int_point.position,
                                    org_ray.dir,
                                    Real(0),
                                    infinity<Real>()};
                        } else {
                            r = Ray{int_point.position, org_ray.dir, Real(0),
                                distance(int_point.position, tgt_point.position)};
                        }
                        const auto &medium = scene.mediums[int_medium_id];
                        tr1 = transmittance(medium, r);
                    }
                    return tr0 * tr1;
                } else {
                    // Block by something
                    return Vector3{0, 0, 0};
                }
            }
        }
    } else {
        auto r = Ray{};
        if (!tgt_isect.valid()) {
            r = Ray{org_ray.org, org_ray.dir, Real(0), infinity<Real>()};
        } else {
            r = Ray{org_ray.org, org_ray.dir, Real(0),
                    distance(org_ray.org, tgt_point.position)};
        }
        if (medium_id != -1) {
            const auto &medium = scene.mediums[medium_id];
            return transmittance(medium, r);
        } else {
            // vaccum
            return Vector3{1, 1, 1};
        }
    }
}

DEVICE
inline
void d_eval_transmittance(const FlattenScene &scene,
                          const Intersection &int_isect,
                          const Ray &org_ray,
                          const Ray &int_ray,
                          const int medium_id,
                          const int int_medium_id,
                          const SurfacePoint &int_point,
                          const Intersection &tgt_isect,
                          const SurfacePoint &tgt_point,
                          const Vector3 &d_output,
                          DMedium *d_mediums,
                          Vector3 &d_org_p,
                          Vector3 &d_int_p,
                          Vector3 &d_tgt_p) {
    if (int_isect.valid()) {
        const auto &shape = scene.shapes[int_isect.shape_id];
        if (int_isect == tgt_isect) {
            // There is no intermediate intersect
            if (medium_id != -1) {
                const auto &medium = scene.mediums[medium_id];
                auto r = Ray{org_ray.org, org_ray.dir, Real(0),
                    distance(org_ray.org, tgt_point.position)};
                // return transmittance(medium, dist);
                auto d_r = Ray::zero();
                d_transmittance(medium,
                                r,
                                d_output,
                                d_mediums[medium_id],
                                d_r);
                auto d_tgt_p_ = Vector3{0, 0, 0};
                d_distance(org_ray.org,
                           tgt_point.position,
                           d_r.tmax,
                           d_org_p,
                           d_tgt_p_);
                d_org_p += d_r.org;
                // int == tgt
                d_tgt_p += d_tgt_p_;
                d_int_p += d_tgt_p_;
            } else {
                return;
            }
        } else {
            if (shape.material_id != -1) {
                // Hit opaque object, transmittance = 0
                return;
            } else {
                // Hit media bounary
                // tmax <= 0 means the ray is blocked
                if (isfinite(int_ray.tmax) || int_ray.tmax > 0) {
                    auto tr0 = Vector3{1, 1, 1};
                    auto tr1 = Vector3{1, 1, 1};
                    auto r0 = Ray{};
                    auto r1 = Ray{};
                    if (medium_id != -1) {
                        const auto &medium = scene.mediums[medium_id];
                        r0 = Ray{int_point.position, org_ray.dir, Real(0),
                                 distance(org_ray.org, int_point.position)};
                        tr0 = transmittance(medium, r0);
                    }
                    if (int_medium_id != -1) {
                        if (!tgt_isect.valid()) {
                            r1 = Ray{int_point.position,
                                     org_ray.dir,
                                     Real(0),
                                     infinity<Real>()};
                        } else {
                            r1 = Ray{int_point.position, org_ray.dir, Real(0),
                                     distance(int_point.position, tgt_point.position)};
                        }
                        const auto &medium = scene.mediums[int_medium_id];
                        tr1 = transmittance(medium, r1);
                    }
                    // return tr0 * tr1;
                    auto d_tr0 = d_output * tr1;
                    auto d_tr1 = d_output * tr0;
                    if (medium_id != -1) {
                        const auto &medium = scene.mediums[medium_id];
                        // tr0 = transmittance(medium, r0)
                        auto d_r0 = Ray::zero();
                        d_transmittance(medium,
                                        r0,
                                        d_tr0,
                                        d_mediums[medium_id],
                                        d_r0);
                        d_org_p += d_r0.org;
                        d_distance(org_ray.org,
                                   int_point.position,
                                   d_r0.tmax,
                                   d_org_p,
                                   d_int_p);
                    }
                    if (int_medium_id != -1) {
                        const auto &medium = scene.mediums[int_medium_id];
                        // tr1 = transmittance(medium, r1)
                        auto d_r1 = Ray::zero();
                        d_transmittance(medium,
                                        r1,
                                        d_tr1,
                                        d_mediums[int_medium_id],
                                        d_r1);
                        d_int_p += d_r1.org;
                        d_distance(int_point.position,
                                   tgt_point.position,
                                   d_r1.tmax,
                                   d_int_p,
                                   d_tgt_p);
                    }
                } else {
                    // Block by something
                    return;
                }
            }
        }
    } else {
        if (medium_id != -1) {
            const auto &medium = scene.mediums[medium_id];
            auto r = Ray{};
            if (!tgt_isect.valid()) {
                r = Ray{org_ray.org, org_ray.dir, Real(0), infinity<Real>()};
            } else {
                r = Ray{org_ray.org, org_ray.dir, Real(0),
                        distance(org_ray.org, tgt_point.position)};
            }
            // return transmittance(medium, dist);
            auto d_r = Ray::zero();
            d_transmittance(medium,
                            r,
                            d_output,
                            d_mediums[medium_id],
                            d_r);
            d_org_p += d_r.org;
            d_distance(org_ray.org,
                       tgt_point.position,
                       d_r.tmax,
                       d_org_p,
                       d_tgt_p);
        } else {
            // vaccum
            return;
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
        const auto &next_isect = next_isects[pixel_id];
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
        if (light_ray.tmax > 0) { // tmax < 0 means the ray is blocked
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
                        auto scatter_val = Vector3{1, 1, 1};
                        auto scatter_pdf = Real(1);
                        if (eval_phase) {
                            auto phase_function =
                                get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                            auto p = phase_function_eval(phase_function, wi, wo);
                            scatter_val = Vector3{p, p, p};
                            scatter_pdf = p; // We assume perfect importance sampling
                        } else {
                            // On surface
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            if (surface_shape.material_id >= 0) {
                                const auto &material = scene.materials[surface_shape.material_id];
                                scatter_val = bsdf(material, surface_point, wi, wo, min_rough);
                                scatter_pdf =
                                    bsdf_pdf(material, surface_point, wi, wo, min_rough);
                            }
                        }
                        auto mis_weight =
                            Real(1 / (1 + square((double)scatter_pdf * geometry_term / (double)pdf_nee)));
                        nee_contrib =
                            (mis_weight * geometry_term / pdf_nee) * scatter_val * light_contrib;
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
                    auto scatter_val = Vector3{1, 1, 1};
                    auto scatter_pdf = Real(1);
                    if (eval_phase) {
                        auto phase_function =
                            get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                        auto p = phase_function_eval(phase_function, wi, wo);
                        scatter_val = Vector3{p, p, p};
                        scatter_pdf = p; // We assume perfect importance sampling
                    } else {
                        const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                        if (surface_shape.material_id >= 0) {
                            const auto &material = scene.materials[surface_shape.material_id];
                            scatter_val = bsdf(material, surface_point, wi, wo, min_rough);
                            scatter_pdf =
                                bsdf_pdf(material, surface_point, wi, wo, min_rough);
                        }
                    }
                    auto mis_weight = Real(1 / (1 + square((double)scatter_pdf / (double)pdf_nee)));
                    nee_contrib = (mis_weight / pdf_nee) * scatter_val * light_contrib;
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
                    } else if (scatter_isect == next_isect) {
                        const auto &scatter_shape = scene.shapes[scatter_isect.shape_id];
                        if (scatter_shape.material_id == -1 && scene.envmap != nullptr) {
                            // We see the environment map through an index-matched medium
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
        // Scatter transmittance
        if (medium_ids != nullptr) {
            // Only need to evaluate when scatter ray hits a light source
            auto need_eval_transmittance = false;
            if (scatter_isect.valid()) {
                const auto &light_shape = scene.shapes[scatter_isect.shape_id];
                if (light_shape.light_id >= 0) {
                    need_eval_transmittance = true;
                } else {
                    if (scatter_isect == next_isect) {
                        const auto &scatter_shape = scene.shapes[scatter_isect.shape_id];
                        if (scatter_shape.material_id == -1 && scene.envmap != nullptr) {
                            need_eval_transmittance = true;
                        }
                    }
                }
            } else {
                if (scene.envmap != nullptr) {
                    need_eval_transmittance = true;
                }
            }
            if (need_eval_transmittance) {
                scatter_contrib *= eval_transmittance(
                        scene,
                        next_isect,
                        scatter_ray,
                        scatter_int_rays[pixel_id],
                        medium_ids[pixel_id],
                        scatter_medium_ids[pixel_id],
                        next_points[pixel_id],
                        scatter_isect,
                        scatter_point);
            }
        }

        if (!isfinite(nee_contrib)) {
            nee_contrib = Vector3{0, 0, 0};
        }

        if (!isfinite(scatter_contrib)) {
            nee_contrib = Vector3{0, 0, 0};
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
        const auto &scatter_ray = scatter_rays[pixel_id];
        const auto &scatter_ray_differential =
            scatter_ray_differentials[pixel_id];
        const auto &min_rough = min_roughness[pixel_id];
        const auto &scatter_isect = scatter_isects[pixel_id];
        const auto &scatter_point = scatter_points[pixel_id];
        const auto &next_isect = scatter_isects[pixel_id];
        // const auto &next_point = next_points[pixel_id];
        const auto &d_next_throughput = d_next_throughputs[pixel_id];
        const auto &d_next_ray = d_next_rays[pixel_id];
        const auto &d_next_ray_differential = d_next_ray_differentials[pixel_id];
        const auto &d_next_point = d_next_points[pixel_id];

        auto &d_throughput = d_throughputs[pixel_id];
        auto &d_incoming_ray = d_incoming_rays[pixel_id];
        auto &d_surface_point = d_surface_points[pixel_id];

        auto wi = -incoming_ray.dir;
        auto p = surface_point.position;
        if (medium_ids != nullptr && medium_distances != nullptr) {
            if (medium_ids[pixel_id] > 0) {
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
        d_surface_point = SurfacePoint::zero();
        auto path_transmittance = Vector3{1, 1, 1};
        if (path_transmittances != nullptr) {
            path_transmittance = path_transmittances[pixel_id];
        }
        auto d_path_transmittance = Vector3{0, 0, 0};

        auto d_p = Vector3{0, 0, 0};

        // Next event estimation
        if (light_ray.tmax > 0) { // tmax < 0 means the ray is blocked
            auto nee_transmittance = Vector3{1, 1, 1};
            if (medium_ids != nullptr) {
                nee_transmittance =
                    eval_transmittance(scene,
                                       light_int_isects[pixel_id],
                                       light_ray,
                                       light_int_rays[pixel_id],
                                       medium_ids[pixel_id],
                                       light_medium_ids[pixel_id],
                                       light_int_points[pixel_id],
                                       light_isect,
                                       light_points[pixel_id]);
            }
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

                        auto scatter_val = Vector3{1, 1, 1};
                        auto scatter_pdf = Real(1);
                        if (eval_phase) {
                            auto phase_function =
                                get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                            auto p = phase_function_eval(phase_function, wi, wo);
                            scatter_val = Vector3{p, p, p};
                            scatter_pdf = p; // assume perfect importance sampling
                        } else {
                            // On surface
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            if (surface_shape.material_id >= 0) {
                                const auto &material = scene.materials[surface_shape.material_id];
                                scatter_val = bsdf(material, surface_point, wi, wo, min_rough);
                                scatter_pdf =
                                    bsdf_pdf(material, surface_point, wi, wo, min_rough);
                            }
                        }
                        auto mis_weight =
                            Real(1 / (1 + square((double)scatter_pdf * geometry_term / (double)pdf_nee)));
                        auto nee_contrib =
                            (mis_weight * geometry_term / pdf_nee) *
                            scatter_val * light_contrib * nee_transmittance;

                        auto d_nee_contrib = d_path_contrib * throughput * path_transmittance;
                        d_throughput += d_path_contrib * nee_contrib * path_transmittance;
                        if (path_transmittances != nullptr) {
                            d_path_transmittance += d_path_contrib * throughput * nee_contrib;
                        }
                        auto weight = mis_weight / pdf_nee;
                        // nee_contrib = (weight * geometry_term)
                        //                scatter_val * light_contrib * nee_transmittance
                        // Ignore derivatives of MIS weight & PMF
                        auto d_weight = geometry_term *
                            sum(d_nee_contrib * scatter_val * light_contrib * nee_transmittance);
                        auto d_geometry_term =
                            weight * sum(d_nee_contrib * scatter_val * light_contrib * nee_transmittance);
                        auto d_light_contrib =
                            weight * d_nee_contrib * geometry_term * scatter_val * nee_transmittance;
                        auto d_scatter_val =
                            weight * d_nee_contrib * geometry_term * light_contrib * nee_transmittance;

                        auto d_light_point = SurfacePoint::zero();
                        if (medium_ids != nullptr) {
                            // Backprop nee transmittance
                            auto d_nee_transmittance =
                                (weight * geometry_term * scatter_val) *
                                sum(d_nee_contrib * light_contrib);
                            const auto &light_int_isect = light_int_isects[pixel_id];
                            auto d_int_p = Vector3{0, 0, 0};
                            d_eval_transmittance(scene,
                                                 light_int_isect,
                                                 light_ray,
                                                 light_int_rays[pixel_id],
                                                 medium_ids[pixel_id],
                                                 light_medium_ids[pixel_id],
                                                 light_int_points[pixel_id],
                                                 light_isect,
                                                 light_point,
                                                 d_nee_transmittance,
                                                 d_mediums,
                                                 d_p,
                                                 d_int_p,
                                                 d_light_point.position);
                            if (light_int_isect.shape_id >= 0) {
                                const auto &int_shape = scene.shapes[light_int_isect.shape_id];
                                // int_p = intersect(light_ray, light_int_isect)
                                Vector3 d_int_v_p[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                                Vector3 d_int_v_n[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                                Vector2 d_int_v_uv[3] = {Vector2{0, 0}, Vector2{0, 0}};
                                Vector3 d_int_v_c[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                                auto d_ray = DRay();
                                auto d_next_ray_differential = RayDifferential::zero();
                                auto d_point = SurfacePoint::zero();
                                auto d_ray_differential = RayDifferential::zero();
                                d_point.position = d_int_p;
                                d_intersect_shape(int_shape,
                                                  light_int_isect.tri_id,
                                                  light_int_rays[pixel_id],
                                                  RayDifferential::zero(),
                                                  d_point,
                                                  d_next_ray_differential,
                                                  d_ray,
                                                  d_ray_differential,
                                                  d_int_v_p,
                                                  d_int_v_n,
                                                  d_int_v_uv,
                                                  d_int_v_c);
                                // ray = Ray{p, normalize(light_point.position - p)}
                                d_p += d_ray.org;
                                auto d_unnorm_dir = d_normalize(light_point.position - p, d_ray.dir);
                                d_p -= d_unnorm_dir;
                                d_light_point.position += d_unnorm_dir;
                                // Accumulate derivatives
                                // n & uv & c has no impact
                                auto int_tri_index = get_indices(int_shape, light_int_isect.tri_id);
                                atomic_add(&d_shapes[light_int_isect.shape_id].vertices[
                                    3 * int_tri_index[0]], d_int_v_p[0]);
                                atomic_add(&d_shapes[light_int_isect.shape_id].vertices[
                                    3 * int_tri_index[1]], d_int_v_p[1]);
                                atomic_add(&d_shapes[light_int_isect.shape_id].vertices[
                                    3 * int_tri_index[2]], d_int_v_p[2]);
                            }
                        }

                        auto d_wi = Vector3{0, 0, 0};
                        auto d_wo = Vector3{0, 0, 0};
                        if (eval_phase) {
                            // Compute the phase function pdf
                            auto phase_function =
                                get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                            // phase_val = phase_function_eval(
                            //     phase_function, wi, wo)
                            auto d_phase_val = sum(d_scatter_val);

                            // phase_val = phase_function_eval(phase_function, wi, wo)
                            auto d_phase_function = DPhaseFunction(phase_function);
                            d_phase_function_eval(phase_function, wi, wo, d_phase_val,
                                d_phase_function, d_wi, d_wo);
                            // phase_function =
                            //     get_phase_function(scene.mediums[medium_ids[pixel_id]])
                            d_get_phase_function(scene.mediums[medium_ids[pixel_id]],
                                                 d_phase_function,
                                                 d_mediums[medium_ids[pixel_id]]);
                        } else {
                            // Not in a medium
                            // bsdf_val = bsdf(material, surface_point, wi, wo, min_rough)
                            if (shading_shape.material_id >= 0) {
                                d_bsdf(material, surface_point, wi, wo, min_rough, d_scatter_val,
                                       d_material, d_surface_point, d_wi, d_wo);
                            }
                        }

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
                        d_surface_point.position -= d_dir;
                        // wi = -incoming_ray.dir
                        d_incoming_ray.dir -= d_wi;

                        // sample point on light
                        d_sample_shape(light_shape, light_isect.tri_id,
                            light_sample.uv, d_light_point, d_light_vertices);

                        // Accumulate derivatives
                        auto light_tri_index = get_indices(light_shape, light_isect.tri_id);
                        atomic_add(&d_shapes[light_isect.shape_id].vertices[
                            3 * light_tri_index[0]], d_light_vertices[0]);
                        atomic_add(&d_shapes[light_isect.shape_id].vertices[
                            3 * light_tri_index[1]], d_light_vertices[1]);
                        atomic_add(&d_shapes[light_isect.shape_id].vertices[
                            3 * light_tri_index[2]], d_light_vertices[2]);
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
                    auto scatter_val = Vector3{1, 1, 1};
                    auto scatter_pdf = Real(1);
                    if (eval_phase) {
                        auto phase_function =
                            get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                        auto p = phase_function_eval(phase_function, wi, wo);
                        scatter_val = Vector3{p, p, p};
                        scatter_pdf = p; // We assume perfect importance sampling
                    } else {
                        const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                        if (surface_shape.material_id >= 0) {
                            const auto &material = scene.materials[surface_shape.material_id];
                            scatter_val = bsdf(material, surface_point, wi, wo, min_rough);
                            scatter_pdf =
                                bsdf_pdf(material, surface_point, wi, wo, min_rough);
                        }
                    }
                    auto mis_weight = Real(1 / (1 + square((double)scatter_pdf / (double)pdf_nee)));
                    auto nee_contrib = (mis_weight / pdf_nee) * scatter_val * light_contrib * nee_transmittance;

                    // path_contrib = path_transmittance * throughput * (nee_contrib + scatter_contrib)
                    auto d_nee_contrib = d_path_contrib * throughput * path_transmittance;
                    d_throughput += d_path_contrib * nee_contrib * path_transmittance;
                    if (path_transmittances != nullptr) {
                        d_path_transmittance += d_path_contrib * throughput * nee_contrib;
                    }
                    auto weight = mis_weight / pdf_nee;
                    // nee_contrib = weight * light_contrib * scatter_val * nee_transmittance
                    auto d_light_contrib = weight * d_nee_contrib * scatter_val * nee_transmittance;
                    auto d_scatter_val = weight * d_nee_contrib * nee_transmittance;
                    if (medium_ids != nullptr) {
                        // Backprop nee transmittance
                        auto d_nee_transmittance =
                            (weight * scatter_val) * sum(d_nee_contrib * light_contrib);
                        const auto &light_int_isect = light_int_isects[pixel_id];
                        auto d_int_p = Vector3{0, 0, 0};
                        auto d_tgt_p = Vector3{0, 0, 0};
                        d_eval_transmittance(scene,
                                             light_int_isect,
                                             light_ray,
                                             light_int_rays[pixel_id],
                                             medium_ids[pixel_id],
                                             light_medium_ids[pixel_id],
                                             light_int_points[pixel_id],
                                             light_isect,
                                             light_points[pixel_id],
                                             d_nee_transmittance,
                                             d_mediums,
                                             d_p,
                                             d_int_p,
                                             d_tgt_p);
                        if (light_int_isect.shape_id >= 0) {
                            const auto &int_shape = scene.shapes[light_int_isect.shape_id];
                            // int_p = intersect(light_ray, light_int_isect)
                            Vector3 d_int_v_p[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                            Vector3 d_int_v_n[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                            Vector2 d_int_v_uv[3] = {Vector2{0, 0}, Vector2{0, 0}};
                            Vector3 d_int_v_c[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                            auto d_ray = DRay();
                            auto d_next_ray_differential = RayDifferential::zero();
                            auto d_point = SurfacePoint::zero();
                            auto d_ray_differential = RayDifferential::zero();
                            d_point.position = d_int_p;
                            d_intersect_shape(int_shape,
                                              light_int_isect.tri_id,
                                              light_int_rays[pixel_id],
                                              RayDifferential::zero(),
                                              d_point,
                                              d_next_ray_differential,
                                              d_ray,
                                              d_ray_differential,
                                              d_int_v_p,
                                              d_int_v_n,
                                              d_int_v_uv,
                                              d_int_v_c);
                            // ray = Ray{p, envmap_sample(...)}.
                            // We don't backprop to envmap_sample:
                            //    envmap_pdf derivatives can contain high variance
                            d_p += d_ray.org;
                            // Accumulate derivatives
                            // n & uv & c has no impact
                            auto int_tri_index = get_indices(int_shape, light_int_isect.tri_id);
                            atomic_add(&d_shapes[light_int_isect.shape_id].vertices[
                                3 * int_tri_index[0]], d_int_v_p[0]);
                            atomic_add(&d_shapes[light_int_isect.shape_id].vertices[
                                3 * int_tri_index[1]], d_int_v_p[1]);
                            atomic_add(&d_shapes[light_int_isect.shape_id].vertices[
                                3 * int_tri_index[2]], d_int_v_p[2]);
                        }
                    }

                    auto d_wi = Vector3{0, 0, 0};
                    auto d_wo = Vector3{0, 0, 0};
                    if (eval_phase) {
                        // Compute the phase function pdf
                        auto phase_function =
                            get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                        // phase_val = phase_function_eval(
                        //     phase_function, wi, wo)
                        auto d_phase_val = sum(d_scatter_val);

                        // phase_val = phase_function_eval(phase_function, wi, wo)
                        auto d_phase_function = DPhaseFunction(phase_function);
                        d_phase_function_eval(phase_function, wi, wo, d_phase_val,
                            d_phase_function, d_wi, d_wo);
                        // phase_function =
                        //     get_phase_function(scene.mediums[medium_ids[pixel_id]])
                        d_get_phase_function(scene.mediums[medium_ids[pixel_id]],
                                             d_phase_function,
                                             d_mediums[medium_ids[pixel_id]]);
                    } else {
                        // Not in a medium
                        // bsdf_val = bsdf(material, surface_point, wi, wo, min_rough)
                        if (shading_shape.material_id >= 0) {
                            d_bsdf(material, surface_point, wi, wo, min_rough, d_scatter_val,
                                   d_material, d_surface_point, d_wi, d_wo);
                        }
                    }

                    auto d_ray_diff = RayDifferential{
                        Vector3{0, 0, 0}, Vector3{0, 0, 0},
                        Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                    // light_contrib = eval_envmap(*scene.envmap, wo, ray_diff)
                    d_envmap_eval(*scene.envmap, wo, ray_diff, d_light_contrib,
                        *d_envmap, d_wo, d_ray_diff);

                    // We don't propagate to pdf_nee: envmap_pdf derivatives can contain high variance

                    // wi = -incoming_ray.dir;
                    d_incoming_ray.dir -= d_wi;
                }
            }
        }

        // BSDF or phase importance sampling
        auto scatter_transmittance = Vector3{1, 1, 1};
        if (medium_ids != nullptr) {
            // Only need to evaluate when scatter ray hits a light source
            auto need_eval_transmittance = false;
            if (scatter_isect.valid()) {
                const auto &light_shape = scene.shapes[scatter_isect.shape_id];
                if (light_shape.light_id >= 0) {
                    need_eval_transmittance = true;
                } else {
                    if (scatter_isect == next_isect) {
                        const auto &scatter_shape = scene.shapes[scatter_isect.shape_id];
                        if (scatter_shape.material_id == -1 && scene.envmap != nullptr) {
                            need_eval_transmittance = true;
                        }
                    }
                }
            } else {
                if (scene.envmap != nullptr) {
                    need_eval_transmittance = true;
                }
            }
            if (need_eval_transmittance) {
                scatter_transmittance =
                    eval_transmittance(scene,
                                       next_isects[pixel_id],
                                       scatter_ray,
                                       scatter_int_rays[pixel_id],
                                       medium_ids[pixel_id],
                                       scatter_medium_ids[pixel_id],
                                       next_points[pixel_id],
                                       scatter_isect,
                                       scatter_point);
            }
        }

        if (scatter_isect.valid()) {
            // We hit a surface
            const auto &light_shape = scene.shapes[scatter_isect.shape_id];
            if (distance_squared(scatter_point.position, p) > 1e-20f) {
                auto wo = scatter_ray.dir;
                auto scatter_val = Vector3{1, 1, 1};
                auto scatter_pdf = Real(1);
                if (eval_phase) {
                    // Inside a medium
                    auto phase_function =
                        get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                    auto phase_eval = phase_function_eval(phase_function, wi, wo);
                    // We assume perfect importance sampling
                    scatter_val = Vector3{phase_eval, phase_eval, phase_eval};
                    scatter_pdf = phase_eval;
                } else {
                    // On a surface
                    const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                    if (surface_shape.material_id >= 0) {
                        const auto &material = scene.materials[surface_shape.material_id];
                        scatter_val = bsdf(material, surface_point, wi, wo, min_rough);
                        scatter_pdf = bsdf_pdf(material, surface_point, wi, wo, min_rough);
                    }
                }
                if (scatter_pdf > 1e-20f) {
                    // next_throughput = path_transmittance * throughput * (scatter_val / scatter_pdf)
                    if (path_transmittances != nullptr) {
                        d_path_transmittance +=
                            d_next_throughput * throughput * (scatter_val / scatter_pdf);
                    }
                    d_throughput += d_next_throughput * path_transmittance * (scatter_val / scatter_pdf);
                    auto d_scatter_val = d_next_throughput * path_transmittance / scatter_pdf;
                    auto d_next_point_ = d_next_point;
                    auto d_wi = Vector3{0, 0, 0};
                    auto d_wo = d_next_ray.dir; // propagated from the next intersection

                    if (light_shape.light_id >= 0) {
                        const auto &light = scene.area_lights[light_shape.light_id];
                        if (light.two_sided || dot(-wo, scatter_point.shading_frame.n) > 0) {
                            auto light_contrib = light.intensity;
                            auto light_pmf = scene.light_pmf[light_shape.light_id];
                            auto light_area = scene.light_areas[light_shape.light_id];
                            auto inv_area = 1 / light_area;
                            auto geometry_term = fabs(dot(wo, scatter_point.geom_normal)) /
                                distance_squared(scatter_point.position, p);
                            auto pdf_nee = (light_pmf * inv_area) / geometry_term;
                            auto mis_weight = Real(1 / (1 + square((double)pdf_nee / (double)scatter_pdf)));
                            auto scatter_contrib = (mis_weight / scatter_pdf) *
                                scatter_val * light_contrib * scatter_transmittance;

                            // path_contrib = path_transmittance * throughput * (nee_contrib + scatter_contrib)
                            auto d_scatter_contrib = d_path_contrib * throughput * path_transmittance;
                            d_throughput += d_path_contrib * scatter_contrib * path_transmittance;
                            if (path_transmittances != nullptr) {
                                d_path_transmittance += d_path_contrib * throughput * scatter_contrib;
                            }

                            // scatter_contrib = (mis_weight / scatter_pdf) *
                            //     scatter_val * light_contrib * scatter_transmittance
                            auto weight = mis_weight / scatter_pdf;
                            d_scatter_val += weight * d_scatter_contrib * scatter_val *
                                light_contrib * scatter_transmittance;
                            auto d_light_contrib = weight * d_scatter_contrib *
                                scatter_val * scatter_transmittance;
                            if (medium_ids != nullptr) {
                                // Backprop scatter transmittance
                                auto d_scatter_transmittance = weight *
                                    d_scatter_contrib * scatter_val * light_contrib;
                                auto d_tgt_p = Vector3{0, 0, 0};
                                d_eval_transmittance(scene,
                                                     next_isects[pixel_id],
                                                     scatter_ray,
                                                     scatter_int_rays[pixel_id],
                                                     medium_ids[pixel_id],
                                                     scatter_medium_ids[pixel_id],
                                                     next_points[pixel_id],
                                                     scatter_isect,
                                                     scatter_point,
                                                     d_scatter_transmittance,
                                                     d_mediums,
                                                     d_p,
                                                     d_next_point_.position,
                                                     d_tgt_p);
                                if (next_isect != scatter_isect) {
                                    // Scatter ray hits two surfaces, we only propagate the second hit here.
                                    // The first hit will be propagated later since we always have that even
                                    // without participating media.
                                    if (scatter_isect.shape_id >= 0) {
                                        const auto &scatter_shape = scene.shapes[scatter_isect.shape_id];
                                        // scatter_p = intersect(scatter_int_rays, scatter_isect)
                                        Vector3 d_scatter_v_p[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                                        Vector3 d_scatter_v_n[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                                        Vector2 d_scatter_v_uv[3] = {Vector2{0, 0}, Vector2{0, 0}};
                                        Vector3 d_scatter_v_c[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                                        auto d_ray = DRay();
                                        auto d_next_ray_differential = RayDifferential::zero();
                                        auto d_point = SurfacePoint::zero();
                                        auto d_ray_differential = RayDifferential::zero();
                                        d_point.position = d_tgt_p;
                                        d_intersect_shape(scatter_shape,
                                                          scatter_isect.tri_id,
                                                          scatter_int_rays[pixel_id],
                                                          RayDifferential::zero(),
                                                          d_point,
                                                          d_next_ray_differential,
                                                          d_ray,
                                                          d_ray_differential,
                                                          d_scatter_v_p,
                                                          d_scatter_v_n,
                                                          d_scatter_v_uv,
                                                          d_scatter_v_c);
                                        // ray = Ray{int_p, bsdf_sample(......)}
                                        d_next_point_.position += d_ray.org;
                                        // We don't propagate through bsdf_sample due to the occasional high variance
                                        // For simplicity we also don't propagate through phase function sampling
                                        // Accumulate derivatives
                                        // n & uv & c has no impact
                                        auto scatter_tri_index =
                                            get_indices(scatter_shape, scatter_isect.tri_id);
                                        atomic_add(&d_shapes[scatter_isect.shape_id].vertices[
                                            3 * scatter_tri_index[0]], d_scatter_v_p[0]);
                                        atomic_add(&d_shapes[scatter_isect.shape_id].vertices[
                                            3 * scatter_tri_index[1]], d_scatter_v_p[1]);
                                        atomic_add(&d_shapes[scatter_isect.shape_id].vertices[
                                            3 * scatter_tri_index[2]], d_scatter_v_p[2]);
                                    }
                                }
                            }
                            // light_contrib = light.intensity
                            atomic_add(d_area_lights[light_shape.light_id].intensity, d_light_contrib);
                        }
                    } else if (scatter_isect == next_isect) {
                        const auto &scatter_shape = scene.shapes[scatter_isect.shape_id];
                        if (scatter_shape.material_id == -1 && scene.envmap != nullptr) {
                            // We see the environment map through an index-matched medium
                            // XXX: For now we don't use ray differentials for envmap
                            //      A proper approach might be to use a filter radius based on sampling density?
                            RayDifferential ray_diff{Vector3{0, 0, 0}, Vector3{0, 0, 0},
                                                     Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                            auto light_contrib = envmap_eval(*scene.envmap, wo, ray_diff);
                            auto envmap_id = scene.num_lights - 1;
                            auto light_pmf = scene.light_pmf[envmap_id];
                            auto pdf_nee = envmap_pdf(*scene.envmap, wo) * light_pmf;
                            auto mis_weight = Real(1 / (1 + square((double)pdf_nee / (double)scatter_pdf)));
                            auto scatter_contrib = (mis_weight / scatter_pdf) * scatter_val * light_contrib * scatter_transmittance;

                            // path_contrib = path_transmittance * throughput * (nee_contrib + scatter_contrib)
                            auto d_scatter_contrib = d_path_contrib * throughput * path_transmittance;
                            d_throughput += d_path_contrib * scatter_contrib * path_transmittance;
                            if (path_transmittances != nullptr) {
                                d_path_transmittance += d_path_contrib * throughput * scatter_contrib;
                            }

                            // scatter_contrib = (mis_weight / scatter_pdf) *
                            //     scatter_val * light_contrib * scatter_transmittance
                            auto weight = mis_weight / scatter_pdf;
                            d_scatter_val += weight * d_scatter_contrib * scatter_val *
                                light_contrib * scatter_transmittance;
                            auto d_light_contrib = weight * d_scatter_contrib *
                                scatter_val * scatter_transmittance;
                            if (medium_ids != nullptr) {
                                // Backprop scatter transmittance
                                auto d_scatter_transmittance = weight *
                                    d_scatter_contrib * scatter_val * light_contrib;
                                auto d_tgt_p = Vector3{0, 0, 0};
                                d_eval_transmittance(scene,
                                                     next_isects[pixel_id],
                                                     scatter_ray,
                                                     scatter_int_rays[pixel_id],
                                                     medium_ids[pixel_id],
                                                     scatter_medium_ids[pixel_id],
                                                     next_points[pixel_id],
                                                     scatter_isect,
                                                     scatter_point,
                                                     d_scatter_transmittance,
                                                     d_mediums,
                                                     d_p,
                                                     d_next_point_.position,
                                                     d_tgt_p);
                                // safe to ignore d_tgt_p: it's the same as the intermediate
                                // since scatter_isect == next_isect
                            }
                            // light_contrib = eval_envmap(*scene.envmap, wo, ray_diff)
                            auto d_ray_diff = RayDifferential{
                                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                            d_envmap_eval(*scene.envmap, wo, ray_diff, d_light_contrib,
                                *d_envmap, d_wo, d_ray_diff);
                        }
                    }
                    if (eval_phase) {
                        // We didn't propagate through sample_phase_function, so we have to
                        // propagate through phase_function_eval here
                        auto phase_function =
                            get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                        auto d_phase_val = sum(d_scatter_val);
                        // phase_val = phase_function_eval(phase_function, wi, wo)
                        auto d_phase_function = DPhaseFunction(phase_function);
                        d_phase_function_eval(phase_function, wi, wo, d_phase_val,
                            d_phase_function, d_wi, d_wo);
                        // phase_function =
                        //     get_phase_function(scene.mediums[medium_ids[pixel_id]])
                        d_get_phase_function(scene.mediums[medium_ids[pixel_id]],
                                             d_phase_function,
                                             d_mediums[medium_ids[pixel_id]]);
                    } else {
                        const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                        if (surface_shape.material_id >= 0) {
                            const auto &material = scene.materials[surface_shape.material_id];
                            // XXX: Ignore derivative w.r.t. pdf_bsdf since it causes high variance
                            // when propagating back from many bounces
                            // This is still correct since
                            // E[(\nabla f) / p] = \int (\nabla f) / p * p = \int (\nabla f)
                            // An intuitive way to think about this is that we are dividing the pdfs
                            // and multiplying MIS weights also for our gradient estimator
                            // scatter_val = bsdf(material, surface_point, wi, wo, min_rough)
                            d_bsdf(material, surface_point, wi, wo, min_rough, d_scatter_val,
                                   d_material, d_surface_point, d_wi, d_wo);
                        }
                    }

                    // Scattering intersection
                    const auto &next_isect = next_isects[pixel_id];
                    // Since scatter_isect is valid, next_isect must be valid
                    assert(next_isect.valid());
                    const auto &next_shape = scene.shapes[next_isect.shape_id];
                    DRay d_ray;
                    RayDifferential d_scatter_ray_differential{
                        Vector3{0, 0, 0}, Vector3{0, 0, 0},
                        Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                    Vector3 d_next_v_p[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                    Vector3 d_next_v_n[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                    Vector2 d_next_v_uv[3] = {Vector2{0, 0}, Vector2{0, 0}};
                    Vector3 d_next_v_c[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                    d_intersect_shape(next_shape,
                                      next_isect.tri_id,
                                      Ray{surface_point.position, wo},
                                      scatter_ray_differential,
                                      d_next_point_,
                                      d_next_ray_differential,
                                      d_ray,
                                      d_scatter_ray_differential,
                                      d_next_v_p,
                                      d_next_v_n,
                                      d_next_v_uv,
                                      d_next_v_c);
                    // XXX HACK: diffuse interreflection causes a lot of noise
                    // on position derivatives but has small impact on the final derivatives,
                    // so we ignore them here.
                    // A properer way is to come up with an importance sampling distribution,
                    // or use a lot more samples
                    if (min_rough > 0.01f) {
                        d_surface_point.position += d_ray.org;
                    }
                    d_wo += d_ray.dir;

                    // wi = -incoming_ray.dir
                    d_incoming_ray.dir -= d_wi;

                    // Accumulate derivatives
                    auto next_tri_index = get_indices(next_shape, next_isect.tri_id);
                    atomic_add(&d_shapes[next_isect.shape_id].vertices[
                        3 * next_tri_index[0]], d_next_v_p[0]);
                    atomic_add(&d_shapes[next_isect.shape_id].vertices[
                        3 * next_tri_index[1]], d_next_v_p[1]);
                    atomic_add(&d_shapes[next_isect.shape_id].vertices[
                        3 * next_tri_index[2]], d_next_v_p[2]);
                    if (has_uvs(next_shape)) {
                        auto uv_tri_ind = next_tri_index;
                        if (next_shape.uv_indices != nullptr) {
                            uv_tri_ind = get_uv_indices(next_shape, next_isect.tri_id);
                        }
                        atomic_add(&d_shapes[next_isect.shape_id].uvs[2 * uv_tri_ind[0]],
                            d_next_v_uv[0]);
                        atomic_add(&d_shapes[next_isect.shape_id].uvs[2 * uv_tri_ind[1]],
                            d_next_v_uv[1]);
                        atomic_add(&d_shapes[next_isect.shape_id].uvs[2 * uv_tri_ind[2]],
                            d_next_v_uv[2]);
                    }
                    if (has_shading_normals(next_shape)) {
                        auto normal_tri_ind = next_tri_index;
                        if (next_shape.normal_indices != nullptr) {
                            normal_tri_ind = get_normal_indices(next_shape, next_isect.tri_id);
                        }
                        atomic_add(&d_shapes[next_isect.shape_id].normals[3 * normal_tri_ind[0]],
                            d_next_v_n[0]);
                        atomic_add(&d_shapes[next_isect.shape_id].normals[3 * normal_tri_ind[1]],
                            d_next_v_n[1]);
                        atomic_add(&d_shapes[next_isect.shape_id].normals[3 * normal_tri_ind[2]],
                            d_next_v_n[2]);
                    }
                    if (has_colors(next_shape)) {
                        atomic_add(&d_shapes[next_isect.shape_id].colors[3 * next_tri_index[0]],
                            d_next_v_c[0]);
                        atomic_add(&d_shapes[next_isect.shape_id].colors[3 * next_tri_index[1]],
                            d_next_v_c[1]);
                        atomic_add(&d_shapes[next_isect.shape_id].colors[3 * next_tri_index[2]],
                            d_next_v_c[2]);
                    }
                }
            }
        } else if (scene.envmap != nullptr) {
            // Hit environment map
            auto wo = scatter_ray.dir;
            auto scatter_pdf = Real(1);
            auto scatter_val = Vector3{1, 1, 1};
            if (eval_phase) {
                // Intersection with medium was made. Get PDF of phase function
                scatter_pdf = phase_function_pdf(
                    get_phase_function(scene.mediums[medium_ids[pixel_id]]),
                    wo, wi);
                scatter_val = Vector3{scatter_pdf, scatter_pdf, scatter_pdf};
            } else {
                // Intersection with surface was made. Get BSDF PDF
                const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                const auto &material = scene.materials[surface_shape.material_id];
                scatter_pdf = bsdf_pdf(material, surface_point, wi, wo, min_rough);

                if (scatter_pdf > 1e-20f) {
                    scatter_val = bsdf(material, surface_point, wi, wo, min_rough);
                }
            }

            // wo can be zero if sampling of the BSDF or phase function fails
            if (length_squared(wo) > 0 && scatter_pdf > 1e-20f) {
                auto ray_diff = RayDifferential{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                    Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                auto light_contrib = envmap_eval(*scene.envmap, wo, ray_diff);
                auto envmap_id = scene.num_lights - 1;
                auto light_pmf = scene.light_pmf[envmap_id];
                auto pdf_nee = envmap_pdf(*scene.envmap, wo) * light_pmf;
                auto mis_weight = Real(1 / (1 + square((double)pdf_nee / (double)scatter_pdf)));
                auto scatter_contrib = (mis_weight / scatter_pdf) *
                    scatter_val * light_contrib * scatter_transmittance;

                // next_throughput = path_transmittance * throughput * (scatter_val / scatter_pdf)
                if (path_transmittances != nullptr) {
                    d_path_transmittance +=
                        d_next_throughput * throughput * (scatter_val / scatter_pdf);
                }
                d_throughput += d_next_throughput * path_transmittance * (scatter_val / scatter_pdf);
                auto d_scatter_val = d_next_throughput * path_transmittance / scatter_pdf;

                // path_contrib = path_transmittance * throughput * (nee_contrib + scatter_contrib)
                auto d_scatter_contrib = d_path_contrib * throughput * path_transmittance;
                d_throughput += d_path_contrib * scatter_contrib * path_transmittance;
                if (path_transmittances != nullptr) {
                    d_path_transmittance += d_path_contrib * throughput * scatter_contrib;
                }

                // scatter_contrib = (mis_weight / scatter_pdf) *
                //     scatter_val * light_contrib * scatter_transmittance
                auto weight = mis_weight / scatter_pdf;
                d_scatter_val += weight * d_scatter_contrib * scatter_val *
                    light_contrib * scatter_transmittance;
                auto d_light_contrib = weight * d_scatter_contrib *
                    scatter_val * scatter_transmittance;
                if (medium_ids != nullptr) {
                    // Backprop scatter transmittance
                    auto d_scatter_transmittance = weight *
                        d_scatter_contrib * scatter_val * light_contrib;
                    auto d_int_p = Vector3{0, 0, 0};
                    auto d_tgt_p = Vector3{0, 0, 0};
                    d_eval_transmittance(scene,
                                         next_isects[pixel_id],
                                         scatter_ray,
                                         scatter_int_rays[pixel_id],
                                         medium_ids[pixel_id],
                                         scatter_medium_ids[pixel_id],
                                         next_points[pixel_id],
                                         scatter_isect,
                                         scatter_point,
                                         d_scatter_transmittance,
                                         d_mediums,
                                         d_p,
                                         d_int_p,
                                         d_tgt_p);
                    // It is safe to ignore d_int_p & d_tgt_p: we hit nothing.
                }

                auto d_wo = Vector3{0, 0, 0};
                auto d_ray_diff = RayDifferential{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                    Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                // light_contrib = eval_envmap(*scene.envmap, wo, ray_diff)
                d_envmap_eval(*scene.envmap, wo, ray_diff, d_light_contrib,
                              *d_envmap, d_wo, d_ray_diff);
                auto d_wi = Vector3{0, 0, 0};
                if (eval_phase) {
                    // We didn't propagate through sample_phase_function, so we have to
                    // propagate through phase_function_eval here
                    auto phase_function =
                        get_phase_function(scene.mediums[medium_ids[pixel_id]]);
                    auto d_phase_val = sum(d_scatter_val);
                    // phase_val = phase_function_eval(phase_function, wi, wo)
                    auto d_phase_function = DPhaseFunction(phase_function);
                    d_phase_function_eval(phase_function, wi, wo, d_phase_val,
                        d_phase_function, d_wi, d_wo);
                    // phase_function =
                    //     get_phase_function(scene.mediums[medium_ids[pixel_id]])
                    d_get_phase_function(scene.mediums[medium_ids[pixel_id]],
                                         d_phase_function,
                                         d_mediums[medium_ids[pixel_id]]);
                } else {
                    // XXX: Ignore derivative w.r.t. pdf_bsdf since it causes high variance
                    // when propagating back from many bounces
                    // This is still correct since
                    // E[(\nabla f) / p] = \int (\nabla f) / p * p = \int (\nabla f)
                    // An intuitive way to think about this is that we are dividing the pdfs
                    // and multiplying MIS weights also for our gradient estimator
                    // bsdf_val = bsdf(material, surface_point, wi, wo)
                    if (shading_shape.material_id >= 0) {
                        d_bsdf(material, surface_point, wi, wo, min_rough, d_scatter_val,
                               d_material, d_surface_point, d_wi, d_wo);
                    }
                }

                // wi = -incoming_ray.dir
                d_incoming_ray.dir -= d_wi;
            }
        }

        if (medium_ids != nullptr && medium_distances != nullptr) {
            if (medium_ids[pixel_id] > 0) {
                // p = incoming_ray.org + incoming_ray.dir * medium_distances[pixel_id];
                d_incoming_ray.org += d_p;
                d_incoming_ray.dir += d_p * medium_distances[pixel_id];
                auto d_medium_distance = sum(d_p * incoming_ray.dir);
                // path_transmittances[pixel_id] = sample_medium(
                //     scene.mediums[medium_id],
                //     incoming_ray,
                //     surface_isect,
                //     surface_point,
                //     medium_sample,
                //     &next_medium_id,
                //     &medium_distance);
                auto d_tmax = Real(0);
                d_sample_medium(scene.mediums[medium_ids[pixel_id]],
                                incoming_ray,
                                surface_isect,
                                medium_samples[pixel_id],
                                d_path_transmittance,
                                d_medium_distance,
                                d_mediums[medium_ids[pixel_id]],
                                d_tmax);
                // tmax = distance(incoming_ray.org, surface_point.position)
                d_distance(incoming_ray.org, surface_point.position,
                           d_tmax, d_incoming_ray.org, d_surface_point.position);
            }
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Vector3 *throughputs;
    const Vector3 *path_transmittances;
    // Shading point info
    const Ray *incoming_rays;
    const LightSample *light_samples;
    const ScatterSample *scatter_samples;
    const MediumSample *medium_samples;
    const Intersection *surface_isects;
    const SurfacePoint *surface_points;
    const int *medium_ids;
    const Real *medium_distances;
    // NEE info
    const Intersection *light_isects;
    const SurfacePoint *light_points;
    const Ray *light_rays;
    const Intersection *light_int_isects;
    const SurfacePoint *light_int_points;
    const Ray *light_int_rays;
    const int *light_medium_ids;
    // Scatter info
    const Intersection *scatter_isects;
    const SurfacePoint *scatter_points;
    const Ray *scatter_rays;
    const RayDifferential *scatter_ray_differentials;
    const Intersection *next_isects;
    const SurfacePoint *next_points;
    const Ray *scatter_int_rays;
    const int *scatter_medium_ids;
    // Misc
    const Real *min_roughness;
    const Real weight;
    const ChannelInfo channel_info;
    // d_outputs
    const float *d_rendered_image;
    const Vector3 *d_next_throughputs;
    const DRay *d_next_rays;
    const RayDifferential *d_next_ray_differentials;
    const SurfacePoint *d_next_points;
    // d_inputs
    DShape *d_shapes;
    DMaterial *d_materials;
    DAreaLight *d_area_lights;
    DEnvironmentMap *d_envmap;
    DMedium *d_mediums;
    Vector3 *d_throughputs;
    DRay *d_incoming_rays;
    SurfacePoint *d_surface_points;
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
        // Shading point info
        incoming_rays.begin(),
        surface_isects.begin(),
        surface_points.begin(),
        medium_ids.begin(),
        medium_distances.begin(),
        // NEE info
        light_isects.begin(),
        light_points.begin(),
        light_rays.begin(),
        light_int_isects.begin(),
        light_int_points.begin(),
        light_int_rays.begin(),
        light_medium_ids.begin(),
        // Scatter info
        scatter_isects.begin(),
        scatter_points.begin(),
        scatter_rays.begin(),
        next_isects.begin(),
        next_points.begin(),
        scatter_int_rays.begin(),
        scatter_medium_ids.begin(),
        // Misc
        min_roughness.begin(),
        weight,
        channel_info,
        // Output
        next_throughputs.begin(),
        rendered_image,
        edge_contribs.begin()}, active_pixels.size(), scene.use_gpu);
}

void d_accumulate_path_contribs(const Scene &scene,
                                const BufferView<int> &active_pixels,
                                const BufferView<Vector3> &throughputs,
                                const BufferView<Vector3> &path_transmittances,
                                // Shading info
                                const BufferView<Ray> &incoming_rays,
                                const BufferView<LightSample> &light_samples,
                                const BufferView<ScatterSample> &scatter_samples,
                                const BufferView<MediumSample> &medium_samples,
                                const BufferView<Intersection> &surface_isects,
                                const BufferView<SurfacePoint> &surface_points,
                                const BufferView<int> &medium_ids,
                                const BufferView<Real> &medium_distances,
                                // NEE info
                                const BufferView<Intersection> &light_isects,
                                const BufferView<SurfacePoint> &light_points,
                                const BufferView<Ray> &light_rays,
                                const BufferView<Intersection> &light_int_isects,
                                const BufferView<SurfacePoint> &light_int_points,
                                const BufferView<Ray> &light_int_rays,
                                const BufferView<int> &light_medium_ids,
                                // Scatter info
                                const BufferView<Intersection> &scatter_isects,
                                const BufferView<SurfacePoint> &scatter_points,
                                const BufferView<Ray> &scatter_rays,
                                const BufferView<RayDifferential> &scatter_ray_differentials,
                                const BufferView<Intersection> &next_isects,
                                const BufferView<SurfacePoint> &next_points,
                                const BufferView<Ray> &scatter_int_rays,
                                const BufferView<int> &scatter_medium_ids,
                                // Misc
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
                                BufferView<SurfacePoint> d_surface_points) {
    parallel_for(d_path_contribs_accumulator{
        get_flatten_scene(scene),
        active_pixels.begin(),
        throughputs.begin(),
        path_transmittances.begin(),
        // Shading point info
        incoming_rays.begin(),
        light_samples.begin(),
        scatter_samples.begin(),
        medium_samples.begin(),
        surface_isects.begin(),
        surface_points.begin(),
        medium_ids.begin(),
        medium_distances.begin(),
        // NEE info
        light_isects.begin(),
        light_points.begin(),
        light_rays.begin(),
        light_int_isects.begin(),
        light_int_points.begin(),
        light_int_rays.begin(),
        light_medium_ids.begin(),
        // scatter info
        scatter_isects.begin(),
        scatter_points.begin(),
        scatter_rays.begin(),
        scatter_ray_differentials.begin(),
        next_isects.begin(),
        next_points.begin(),
        scatter_int_rays.begin(),
        scatter_medium_ids.begin(),
        // misc
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
        d_surface_points.begin()},
        active_pixels.size(), scene.use_gpu);
}
