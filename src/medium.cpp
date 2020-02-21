#include "medium.h"
#include "active_pixels.h"
#include "parallel.h"
#include "scene.h"
#include "thrust_utils.h"

// DEVICE
// inline
// Real sample_distance(const Medium &medium, const MediumSample &sample) {
//     if (medium.type == MediumType::homogeneous) {
//         // Sample a channel and distance along the ray
//         auto h = medium.homogeneous;
//         auto channel = min(int(sample.uv[0] * 3), 2);
//         return Real(-log(max(1 - sample.uv[1], Real(1e-20))) / h.sigma_t[channel]);
//     } else {
//         return Real(0);
//     }
// }

// DEVICE
// inline
// void d_sample_distance(const Medium &medium,
//                        const MediumSample &sample,
//                        Real &d_output,
//                        DMedium &d_medium) {
//     if (medium.type == MediumType::homogeneous) {
//         // Sample a channel and distance along the ray
//         auto h = medium.homogeneous;
//         auto channel = min(int(sample.uv[0] * 3), 2);

//         // auto t = Real(-log(max(1 - sample.uv[1], Real(1e-20))) / h.sigma_t[channel]);
//         //auto d_t = 1 / (h.sigma_t[channel] - d_output);
//         auto d_t_sigma_t = log(max(1 - sample.uv[1], Real(1e-20))) / square(h.sigma_t[channel]);

//         auto d_sigma_a = h.sigma_s + d_t_sigma_t;
//         atomic_add(&d_medium.sigma_a[0], d_sigma_a[0]);
//         atomic_add(&d_medium.sigma_a[1], d_sigma_a[1]);
//         atomic_add(&d_medium.sigma_a[2], d_sigma_a[2]);

//         auto d_sigma_s = h.sigma_a + d_t_sigma_t;
//         atomic_add(&d_medium.sigma_s[0], d_sigma_s[0]);
//         atomic_add(&d_medium.sigma_s[1], d_sigma_s[1]);
//         atomic_add(&d_medium.sigma_s[2], d_sigma_s[2]);
//     }
// }

/**
 * Sample the given medium to see if the ray is affected by it. The
 * transmittance is encoded in the returned Vector3.
 */
DEVICE
inline
Vector3 sample(const FlattenScene &scene,
               int medium_id,
               const Ray &ray,
               const Intersection &surface_isect,
               const SurfacePoint &surface_point,
               const MediumSample &sample,
               int *next_medium_id,
               Real *medium_distance) {
    const auto &medium = scene.mediums[medium_id];
    if (medium.type == MediumType::homogeneous) {
        auto h = medium.homogeneous;
        // Sample a distance point along the ray and compare it
        // to tmax to see if we are inside of a medium or not
        auto channel = min(int(sample.uv[0] * 3), 2);
        auto dist = Real(-log(max(1 - sample.uv[1], Real(1e-20))) / h.sigma_t[channel]);
        auto t = min(dist, ray.tmax);
        auto inside_medium = t < ray.tmax;

        // Compute the transmittance and sampling density
        auto tr = exp(-h.sigma_t * min(t, MaxFloat));

        // Return the weighting factor for scattering inside of a homogeneous medium
        auto density = inside_medium ? (h.sigma_t * tr) : tr;
        auto pdf = sum(density) / 3;
        // Update intersection data
        *medium_distance = t;
        if (!inside_medium) {
            // shape_id shouldn't be <0
            // (since inside_medium being false indicates ray.tmax = inf)
            // but just to be safe
            if (surface_isect.shape_id >= 0) {
                const auto &shape = scene.shapes[surface_isect.shape_id];
                if (dot(ray.dir, surface_point.geom_normal) < 0) {
                    *next_medium_id = shape.exterior_medium_id;
                } else {
                    *next_medium_id = shape.interior_medium_id;
                }
            } else {
                *next_medium_id = -1;
            }
        } else {
            *next_medium_id = medium_id;
        }
        return inside_medium ? (tr * h.sigma_s / pdf) : (tr / pdf);
    } else {
        return Vector3{0, 0, 0};
    }
}

/**
 * Evaluate the derivative of the sampling process of the medium. For
 * this we need to find the derivative of the transmittance combined
 * with the PDF of the overall density function.
 */
DEVICE
inline
void d_sample(const Medium &medium,
              const Ray &ray,
              const Intersection &surface_isect,
              const MediumSample &sample,
              const Vector3 &d_output,
              DMedium &d_medium,
              Intersection *medium_isect,
              Vector3 *medium_point,
              DRay &d_ray) {
    // if (medium.type == MediumType::homogeneous) {
    //     auto h = medium.homogeneous;
    //     // Sample a distance within the medium and compare it
    //     // to tmax to see if we are inside of a medium or not
    //     auto dist = sample_distance(medium, sample);
    //     auto t = min(dist, ray.tmax);
    //     auto inside_medium = t < ray.tmax;

    //     // Compute the transmittance and sampling density
    //     auto tr = exp(-h.sigma_t * min(t, MaxFloat));

    //     // Return the weighting factor for scattering inside of a homogeneous medium
    //     // and its derivatives
    //     auto density = inside_medium ? (h.sigma_t * tr) : tr;
    //     auto pdf = sum(density) / 3;

    //     // // Update intersection data
    //     // *medium_point = ray.org + ray.dir * t;

    //     // Calculate the partial derivative of the transmittance with respect to t
    //     // f'/pdf is equivalent to (f/pdf)' in terms of expectation
    //     // auto beta = inside_medium ? (tr * h.sigma_s / pdf) : (tr / pdf);
    //     auto d_beta_tr = inside_medium ? (d_output * h.sigma_s / pdf) : (d_output/ pdf);

    //     if (inside_medium) {
    //         // Only make this add if we are inside a medium. The derivative
    //         // of beta with respect to sigma_s should not change otherwise
    //         // auto beta = tr * sigma_s / pdf;
    //         auto d_beta_sigma_s = tr * d_beta_tr / pdf;
    //         atomic_add(d_medium.sigma_s[0], d_beta_sigma_s[0]);
    //         atomic_add(d_medium.sigma_s[1], d_beta_sigma_s[1]);
    //         atomic_add(d_medium.sigma_s[2], d_beta_sigma_s[2]);
    //     }

    //     // We assume min(t, MaxFloat) will always be less than MaxFloat
    //     // for simplification purposes
    //     // auto tr = exp(-h.sigma_t * min(t, MaxFloat))
    //     auto d_tr_t = h.sigma_t * -exp(-h.sigma_t * d_beta_tr);
    //     auto d_tr_sigma_t = d_beta_tr * -exp(-h.sigma_t * d_beta_tr);

    //     // Need to recover d_sigma_a and d_sigma_s from d_tr_sigma_t since
    //     // sigma_t = sigma_a + sigma_s
    //     auto d_sigma_a = h.sigma_s + d_tr_sigma_t;
    //     atomic_add(d_medium.sigma_a[0], d_sigma_a[0]);
    //     atomic_add(d_medium.sigma_a[1], d_sigma_a[1]);
    //     atomic_add(d_medium.sigma_a[2], d_sigma_a[2]);

    //     auto d_sigma_s = h.sigma_a + d_tr_sigma_t;
    //     atomic_add(d_medium.sigma_s[0], d_sigma_s[0]);
    //     atomic_add(d_medium.sigma_s[1], d_sigma_s[1]);
    //     atomic_add(d_medium.sigma_s[2], d_sigma_s[2]);

    //     //*medium_point = ray.org + ray.dir * t;
    //     // TODO Check how we can d_transmittance
    //     //auto d_medium_point_t = d_output;
    //     d_ray.org += d_tr_t;
    //     d_ray.dir += ray.dir * t;

    //     // auto t = min(dist, ray.tmax)
    //     // This can introduce a discontinuity!
    //     auto d_t = min(d_tr_t, ray.tmax);

    //     // auto dist = sample_distance(medium, sample);
    //     auto d_dist = Real(0);
    //     d_sample_distance(medium, sample, d_dist, d_medium);

    //     // Backpropagate the derivative of the transmittance with respect
    //     // to t_max to the intersection function.
    //     auto ray_diff = RayDifferential {
    //         Vector3{0, 0, 0}, Vector3{0, 0, 0},
    //         Vector3{0, 0, 0}, Vector3{0, 0, 0}
    //     };
    //     auto d_v0 = Vector3{0, 0, 0};
    //     auto d_v1 = Vector3{0, 0, 0};
    //     auto d_v2 = Vector3{0, 0, 0};
    //     auto d_ray_diff = RayDifferential {
    //         Vector3{0, 0, 0}, Vector3{0, 0, 0},
    //         Vector3{0, 0, 0}, Vector3{0, 0, 0}
    //     };
    //     d_intersect(Vector3{0, 0, 0},
    //                 Vector3{0, 0, 0},
    //                 Vector3{0, 0, 0},
    //                 ray,
    //                 ray_diff,
    //                 d_t,
    //                 Vector2{0, 0},
    //                 Vector2{0, 0},
    //                 Vector2{0, 0},
    //                 d_v0,
    //                 d_v1,
    //                 d_v2,
    //                 d_ray,
    //                 d_ray_diff);
    // }
}

struct medium_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &medium_sample = medium_samples[pixel_id];
        const auto &surface_isect = surface_isects[pixel_id];
        const auto &surface_point = surface_points[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        const auto &medium_id = medium_ids[pixel_id];
        auto &next_medium_id = next_medium_ids[pixel_id];
        auto &medium_distance = medium_distances[pixel_id];
        if (medium_id >= 0) {
            // We are inside a medium. Sample a distance and compute transmittance.
            // Also update the intersection point.
            transmittances[pixel_id] = sample(
                scene,
                medium_id,
                incoming_ray,
                surface_isect,
                surface_point,
                medium_sample,
                &next_medium_id,
                &medium_distance);
        } else {
            transmittances[pixel_id] = Vector3{1, 1, 1};
            next_medium_id = -1;
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Intersection *surface_isects;
    const SurfacePoint *surface_points;
    const Ray *incoming_rays;
    const MediumSample *medium_samples;
    const int *medium_ids;
    int *next_medium_ids;
    Real *medium_distances;
    Vector3 *transmittances;
};

void sample_medium(const Scene &scene,
                   const BufferView<int> &active_pixels,
                   const BufferView<Intersection> &surface_isects,
                   const BufferView<SurfacePoint> &surface_points,
                   const BufferView<Ray> &incoming_rays,
                   const BufferView<MediumSample> &medium_samples,
                   const BufferView<int> &medium_ids,
                   BufferView<int> next_medium_ids,
                   BufferView<Real> medium_distances,
                   BufferView<Vector3> transmittances) {
    parallel_for(medium_sampler{
        get_flatten_scene(scene),
        active_pixels.begin(),
        surface_isects.begin(),
        surface_points.begin(),
        incoming_rays.begin(),
        medium_samples.begin(),
        medium_ids.begin(),
        next_medium_ids.begin(),
        medium_distances.begin(),
        transmittances.begin()},
        active_pixels.size(), scene.use_gpu);
}

// struct d_medium_sampler {
//     DEVICE void operator()(int idx) {
//         auto pixel_id = active_pixels[idx];
//         const auto &surface_isect = surface_isects[pixel_id];
//         const auto &incoming_ray = incoming_rays[pixel_id];
//         const auto &medium_sample = medium_samples[pixel_id];
//         const auto &medium_id = medium_ids[pixel_id];
//         auto &medium_point = medium_points[pixel_id];
//         auto &d_medium = d_mediums[medium_ids[pixel_id]];

//         if (medium_id >= 0) {
//             // throughputs[pixel_id] *= transmittance;
//             auto d_transmittance = transmittances[pixel_id] * d_throughputs[pixel_id];
//             auto d_ray = d_rays[pixel_id];
//             d_sample(scene.mediums[medium_id],
//                      incoming_ray,
//                      surface_isect,
//                      medium_sample,
//                      d_transmittance,
//                      d_medium,
//                      d_ray);
//         }
//     }

//     const FlattenScene scene;
//     const int *active_pixels;
//     const Intersection *surface_isects;
//     const Ray *incoming_rays;
//     const MediumSample *medium_samples;
//     const int *medium_ids;
//     const Vector3 *transmittances;
//     const Vector3 *d_throughputs;
//     DMedium *d_mediums;
//     DRay *d_rays;
//     Vector3 *d_transmittances;
//     Intersection *medium_isects;
//     Vector3 *medium_points;
//     Vector3 *throughputs;
// };

void d_sample_medium(const Scene &scene,
                     const BufferView<int> &active_pixels,
                     const BufferView<Intersection> &surface_isects,
                     const BufferView<Ray> &incoming_rays,
                     const BufferView<MediumSample> &medium_samples,
                     const BufferView<Vector3> &transmittances,
                     const BufferView<Vector3> &d_throughputs,
                     DScene *d_scene,
                     BufferView<DRay> &d_rays,
                     BufferView<Vector3> &d_transmittances,
                     BufferView<Intersection> medium_isects,
                     BufferView<Vector3> medium_points,
                     BufferView<Vector3> throughputs) {
    // parallel_for(d_medium_sampler{
    //     get_flatten_scene(scene),
    //     active_pixels.begin(),
    //     surface_isects.begin(),
    //     incoming_rays.begin(),
    //     medium_samples.begin(),
    //     transmittances.begin(),
    //     d_throughputs.begin(),
    //     d_scene->mediums.data,
    //     d_rays.begin(),
    //     d_transmittances.begin(),
    //     medium_isects.begin(),
    //     medium_points.begin(),
    //     throughputs.begin()},
    //     active_pixels.size(), scene.use_gpu);
}

struct transmittance_shadow_ray_spawner {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &int_isect = int_isects[pixel_id];
        if (int_isect.valid()) {
            const auto &shape = shapes[int_isect.shape_id];
            if (shape.material_id != -1) {
                // Hit an opaque object
                // Spawn an invalid ray
                shadow_rays[pixel_id] = Ray{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                };
                int_medium_ids[pixel_id] = -1;
            } else {
                // Hit a transparent object, advance the transmittance ray
                auto p0 = int_points[pixel_id].position;
                auto p1 = light_points[pixel_id].position;
                shadow_rays[pixel_id] = Ray{
                    p0, // org
                    normalize(p1 - p0), // dir
                    Real(1e-3), // tmin
                    (1 - 1e-3f) * distance(p0, p1), // tmax
                };
                // Update medium id
                const auto &p = int_points[pixel_id];
                const auto &shape = shapes[int_isect.shape_id];
                if (dot(shadow_rays[pixel_id].dir, p.geom_normal) < 0) {
                    int_medium_ids[pixel_id] = shape.exterior_medium_id;
                } else {
                    int_medium_ids[pixel_id] = shape.interior_medium_id;
                }
            }
        } else {
            // Hit nothing -- don't need to trace shadow ray
            shadow_rays[pixel_id] = Ray{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
            };
            int_medium_ids[pixel_id] = -1;
        }
    }

    const Shape *shapes;
    const int *active_pixels;
    const Intersection *int_isects;
    const SurfacePoint *int_points;
    const SurfacePoint *light_points;
    int *int_medium_ids;
    Ray *shadow_rays;
};

struct transmittance_ray_spawner {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &int_isect = int_isects[pixel_id];
        if (int_isect.valid()) {
            const auto &shape = shapes[int_isect.shape_id];
            if (shape.material_id != -1) {
                // Hit an opaque object
                // Spawn an invalid ray
                next_rays[pixel_id] = Ray{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                };
                int_medium_ids[pixel_id] = -1;
            } else {
                // Hit a transparent object, advance the transmittance ray
                next_rays[pixel_id] = Ray{
                    int_points[pixel_id].position, // org
                    next_rays[pixel_id].dir,
                    Real(1e-3) // tmin
                };
                // Update medium id
                const auto &p = int_points[pixel_id];
                const auto &shape = shapes[int_isect.shape_id];
                if (dot(next_rays[pixel_id].dir, p.geom_normal) < 0) {
                    int_medium_ids[pixel_id] = shape.exterior_medium_id;
                } else {
                    int_medium_ids[pixel_id] = shape.interior_medium_id;
                }
            }
        } else {
            // Hit nothing -- don't need to trace new ray
            next_rays[pixel_id] = Ray{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
            };
            int_medium_ids[pixel_id] = -1;
        }
    }

    const Shape *shapes;
    const int *active_pixels;
    const Intersection *int_isects;
    const SurfacePoint *int_points;
    int *int_medium_ids;
    Ray *next_rays;
};

struct transmittance_intersection_update {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &int_isect = int_isects[pixel_id];
        if (int_isect.valid()) {
            const auto &shape = shapes[int_isect.shape_id];
            if (shape.material_id != -1) {
                // Hit an opaque object
                // set light_isects to the same intersection
                light_isects[pixel_id] = int_isect;
                light_points[pixel_id] = int_points[pixel_id];
            } else {
                // Hit a transparent object
                // leave light_isects as is
            }
        } else {
            // Hit nothing -- set light_isects to invalid
            light_isects[pixel_id] = Intersection{-1, -1};
        }
    }

    const Shape *shapes;
    const int *active_pixels;
    const Intersection *int_isects;
    const SurfacePoint *int_points;
    Intersection *light_isects;
    SurfacePoint *light_points;
};

void trace_nee_transmittance_rays(const Scene &scene,
                                  const BufferView<int> &active_pixels,
                                  const BufferView<Ray> &outgoing_rays,
                                  const BufferView<int> &medium_ids,
                                  const BufferView<Intersection> &light_isects,
                                  const BufferView<SurfacePoint> &light_points,
                                  BufferView<Ray> int_rays,
                                  BufferView<Intersection> int_isects,
                                  BufferView<SurfacePoint> int_points,
                                  BufferView<int> int_medium_ids,
                                  ThrustCachedAllocator &thrust_alloc,
                                  BufferView<OptiXRay> optix_rays,
                                  BufferView<OptiXHit> optix_hits) {
    // Copy outgoing rays to intermediate rays
    DISPATCH(scene.use_gpu, thrust::copy,
        outgoing_rays.begin(), outgoing_rays.end(),
        int_rays.begin());
    // Intersect with the scene
    intersect(scene,
              active_pixels,
              int_rays,
              BufferView<RayDifferential>(),
              int_isects,
              int_points,
              BufferView<RayDifferential>(),
              optix_rays,
              optix_hits);
    // Update medium ids and spawn transmittance shadow ray
    parallel_for(transmittance_shadow_ray_spawner{
            scene.shapes.data,
            active_pixels.begin(),
            int_isects.begin(),
            int_points.begin(),
            light_points.begin(),
            int_medium_ids.begin(),
            int_rays.begin()
        }, active_pixels.size(), scene.use_gpu);
    // Shadow ray
    occluded(scene, active_pixels, int_rays, optix_rays, optix_hits);
}

void trace_scatter_transmittance_rays(const Scene &scene,
                                      const BufferView<int> &active_pixels,
                                      const BufferView<Ray> &outgoing_rays,
                                      const BufferView<RayDifferential> &ray_differentials,
                                      const BufferView<int> &medium_ids,
                                      BufferView<Intersection> light_isects,
                                      BufferView<SurfacePoint> light_points,
                                      BufferView<Ray> int_rays,
                                      BufferView<RayDifferential> new_ray_differentials,
                                      BufferView<Intersection> int_isects,
                                      BufferView<SurfacePoint> int_points,
                                      BufferView<int> int_medium_ids,
                                      ThrustCachedAllocator &thrust_alloc,
                                      BufferView<OptiXRay> optix_rays,
                                      BufferView<OptiXHit> optix_hits) {
    // Copy outgoing rays to intermediate rays
    DISPATCH(scene.use_gpu, thrust::copy,
        outgoing_rays.begin(), outgoing_rays.end(),
        int_rays.begin());
    // Intersect with the scene
    intersect(scene,
              active_pixels,
              int_rays,
              ray_differentials,
              int_isects,
              int_points,
              new_ray_differentials,
              optix_rays,
              optix_hits);
    // Update medium ids and spawn transmittance shadow ray
    parallel_for(transmittance_ray_spawner{
            scene.shapes.data,
            active_pixels.begin(),
            int_isects.begin(),
            int_points.begin(),
            int_medium_ids.begin(),
            int_rays.begin()
        }, active_pixels.size(), scene.use_gpu);
    // Intersect with the scene again
    intersect(scene,
              active_pixels,
              int_rays,
              BufferView<RayDifferential>(),
              light_isects,
              light_points,
              BufferView<RayDifferential>(),
              optix_rays,
              optix_hits);
    // Update intersection
    parallel_for(transmittance_intersection_update{
            scene.shapes.data,
            active_pixels.begin(),
            int_isects.begin(),
            int_points.begin(),
            light_isects.begin(),
            light_points.begin(),
        }, active_pixels.size(), scene.use_gpu);
}
