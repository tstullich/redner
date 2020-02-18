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

/// Calculate the transmittance of a ray segment given a ray
DEVICE
inline
Vector3 transmittance(const Medium &medium, Real distance) {
    if (medium.type == MediumType::homogeneous) {
        // Use Beer's Law to calculate transmittance
        return exp(-medium.homogeneous.sigma_t * min(distance, MaxFloat));
    } else {
        return Vector3{0, 0, 0};
    }
}

DEVICE
inline
void d_transmittance(const Medium &medium,
                     float distance,
                     const Vector3 &d_output,
                     DMedium &d_medium,
                     float &d_distance) {
    if (medium.type == MediumType::homogeneous) {
        // output = exp(-medium.homogeneous.sigma_t * min(distance, MaxFloat));
        if (distance < MaxFloat) {
            auto output = exp(medium.homogeneous.sigma_t * distance);
            auto d_sigma_t = -d_output * output * distance;
            d_distance += (-sum(d_output * output * medium.homogeneous.sigma_t));
            // sigma_t = sigma_a + sigma_s
            atomic_add(&d_medium.sigma_a[0], d_sigma_t[0]);
            atomic_add(&d_medium.sigma_a[1], d_sigma_t[1]);
            atomic_add(&d_medium.sigma_a[2], d_sigma_t[2]);
            atomic_add(&d_medium.sigma_s[0], d_sigma_t[0]);
            atomic_add(&d_medium.sigma_s[1], d_sigma_t[1]);
            atomic_add(&d_medium.sigma_s[2], d_sigma_t[2]);
        }
    } else {
        return;
    }
}

struct transmittance_evaluator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        auto &transmittance_ray = transmittance_rays[pixel_id];
        const auto &medium = mediums[medium_ids[pixel_id]];
        transmittances[pixel_id] *=
            transmittance(medium, transmittance_ray.tmax);
        // Update medium id
        const auto &isect = surface_isects[pixel_id];
        if (isect.valid()) {
            const auto &p = surface_points[pixel_id];
            const auto &shape = shapes[isect.shape_id];
            if (dot(transmittance_ray.dir, p.geom_normal) < 0) {
                medium_ids[pixel_id] = shape.exterior_medium_id;
            } else {
                medium_ids[pixel_id] = shape.interior_medium_id;
            }
            if (shape.material_id != -1) {
                // Hit an opaque object, occluded
                transmittances[pixel_id] = Vector3{0, 0, 0};
                // Invalidate intersection: don't need to trace anymore
                surface_isects[pixel_id].shape_id = -1;
                medium_ids[pixel_id] = -1;
            } else {
                // Hit a transparent object, advance the transmittance ray
                auto target = transmittance_ray.org +
                              transmittance_ray.dir * transmittance_ray.tmax;
                transmittance_ray.org = p.position;
                transmittance_ray.tmax = distance(target, p.position);
            }
        } else {
            // Hit nothing, we've safely reached the target
            // Invalidate the intersection: don't need to trace anymore
            surface_isects[pixel_id].shape_id = -1;
            medium_ids[pixel_id] = -1;
        }
    }

    const Shape *shapes;
    const Medium *mediums;
    const int *active_pixels;
    const SurfacePoint *surface_points;
    Intersection *surface_isects;
    Ray *transmittance_rays;
    int *medium_ids;
    Vector3 *transmittances;
};

void evaluate_transmittance(const Scene &scene,
                            const BufferView<int> &active_pixels,
                            const BufferView<Ray> &outgoing_rays,
                            const BufferView<int> &medium_ids,
                            BufferView<Vector3> transmittances,
                            TransmittanceBuffer &tr_buffer,
                            ThrustCachedAllocator &thrust_alloc,
                            BufferView<OptiXRay> optix_rays,
                            BufferView<OptiXHit> optix_hits) {
    // Evaluate the transmittance between two points
    // To do this we need to find all the participating media boundaries between
    // the two points with IOR = 1. 
    // This means that we want to go through all surfaces with material_id = -1,
    // and stop at an opaque surface.

    auto transmittance_active_pixels = 
        tr_buffer.active_pixels.view(0, active_pixels.size());
    auto transmittance_rays =
        tr_buffer.rays.view(0, outgoing_rays.size());
    auto transmittance_isects =
        tr_buffer.surface_isects.view(0, outgoing_rays.size());
    auto transmittance_points =
        tr_buffer.surface_points.view(0, outgoing_rays.size());
    auto transmittance_medium_ids =
        tr_buffer.medium_ids.view(0, outgoing_rays.size());
    // Copy the active pixels
    DISPATCH(scene.use_gpu, thrust::copy,
        active_pixels.begin(), active_pixels.end(),
        transmittance_active_pixels.begin());
    // Copy outgoing rays to transmittance rays
    DISPATCH(scene.use_gpu, thrust::copy,
        outgoing_rays.begin(), outgoing_rays.end(),
        transmittance_rays.begin());
    // Copy the medium ids
    DISPATCH(scene.use_gpu, thrust::copy,
        medium_ids.begin(), medium_ids.end(),
        transmittance_medium_ids.begin());
    // Initiailize transmittance
    DISPATCH(scene.use_gpu, thrust::fill,
        transmittances.begin(), transmittances.end(), Vector3{1, 1, 1});
    while (transmittance_active_pixels.size() > 0) {
        // Intersect with the scene
        intersect(scene,
                  transmittance_active_pixels,
                  transmittance_rays,
                  BufferView<RayDifferential>(),
                  transmittance_isects,
                  transmittance_points,
                  BufferView<RayDifferential>(),
                  optix_rays,
                  optix_hits);
        // evaluate transmittance for the segment update medium ids
        parallel_for(transmittance_evaluator{
                scene.shapes.data,
                scene.mediums.data,
                active_pixels.begin(),
                transmittance_points.begin(),
                transmittance_isects.begin(),
                transmittance_rays.begin(),
                transmittance_medium_ids.begin(),
                transmittances.begin()},
            transmittance_active_pixels.size(), scene.use_gpu);
        // Stream compaction: remove paths that didn't hit surfaces 
        // or hit opaque surfaces
        update_active_pixels(transmittance_active_pixels,
                             transmittance_isects,
                             transmittance_medium_ids,
                             transmittance_active_pixels,
                             scene.use_gpu);
    }
}
