#include "medium.h"
#include "parallel.h"
#include "scene.h"
#include "thrust_utils.h"

DEVICE
inline
Real sample_distance(const Medium &medium, const MediumSample &sample) {
    if (medium.type == MediumType::homogeneous) {
        // Sample a channel and distance along the ray
        auto h = medium.homogeneous;
        auto channel = min(int(sample.uv[0] * 3), 2);
        return Real(-log(max(1 - sample.uv[1], Real(1e-20))) / h.sigma_t[channel]);
    } else {
        return Real(0);
    }
}

DEVICE
inline
void d_sample_distance(const Medium &medium,
                       const MediumSample &sample,
                       Real &d_output,
                       DMedium &d_medium) {
    if (medium.type == MediumType::homogeneous) {
        // Sample a channel and distance along the ray
        auto h = medium.homogeneous;
        auto channel = min(int(sample.uv[0] * 3), 2);

        // Derivative of the distance sampling function with respect to the
        // sample which varies. We assume that max()' = 1
        //auto t = Real(-log(max(1 - sample.uv[1], Real(1e-20))) / h.sigma_t[channel]);

        // Backpropagate
        // auto t = Real(-log(max(1 - sample.uv[1], Real(1e-20))) / h.sigma_t[channel]);
        auto d_t = 1 / (h.sigma_t[channel] - d_output);
        auto d_t_sigma_t = log(max(1 - sample.uv[1], Real(1e-20))) / (h.sigma_t[channel] * h.sigma_t[channel]);

        auto d_sigma_a = h.sigma_s + d_t_sigma_t;
        atomic_add(&d_medium.sigma_a[0], d_sigma_a[0]);
        atomic_add(&d_medium.sigma_a[1], d_sigma_a[1]);
        atomic_add(&d_medium.sigma_a[2], d_sigma_a[2]);

        auto d_sigma_s = h.sigma_a + d_t_sigma_t;
        atomic_add(&d_medium.sigma_s[0], d_sigma_s[0]);
        atomic_add(&d_medium.sigma_s[1], d_sigma_s[1]);
        atomic_add(&d_medium.sigma_s[2], d_sigma_s[2]);
    }
}

/**
 * Sample the given medium to see if the ray is affected by it. The
 * transmittance is encoded in the returned Vector3.
 */
DEVICE
inline
Vector3 sample(const Medium &medium,
               const Ray &ray,
               const Intersection &surface_isect,
               const MediumSample &sample,
               Intersection *medium_isect,
               Vector3 *medium_point) {
    if (medium.type == MediumType::homogeneous) {
        auto h = medium.homogeneous;
        // Sample a distance point along the ray and compare it
        // to tmax to see if we are inside of a medium or not
        auto dist = sample_distance(medium, sample);
        auto t = min(dist, ray.tmax);
        auto inside_medium = t < ray.tmax;

        // Compute the transmittance and sampling density
        auto tr = exp(-h.sigma_t * min(t, MaxFloat));

        // Return the weighting factor for scattering inside of a homogeneous medium
        auto density = inside_medium ? (h.sigma_t * tr) : tr;
        auto pdf = sum(density) / 3;
        // Update intersection data
        *medium_point = ray.org + ray.dir * t;
        if (inside_medium) {
            // Set the triangle ID to -1 if we are inside a medium
            medium_isect->tri_id = -1;
        } else {
            // Left our current medium. Update intersection data accordingly
            if (medium_isect->prev_medium_id >= 0) {
                // We previously encountered a medium
                medium_isect->medium_id = medium_isect->prev_medium_id;
                medium_isect->shape_id = medium_isect->prev_shape_id;
                medium_isect->prev_medium_id = -1;
                medium_isect->prev_shape_id = -1;
                medium_isect->tri_id = -1;
            } else {
                // No prior intersection was made. Take the surface intersection
                assert(surface_isect.valid());
                *medium_isect = surface_isect;
            }
        }
        return inside_medium ? (tr * h.sigma_s / pdf) : (tr / pdf);
    } else if (medium.type == MediumType::heterogeneous) {
        return Vector3{0, 0, 0};
    } else {
        return Vector3{0, 0, 0};
    }
}

struct medium_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &medium_sample = medium_samples[pixel_id];
        const auto &surface_isect = surface_isects[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        auto &medium_isect = medium_isects[pixel_id];
        auto &medium_point = medium_points[pixel_id];
        if (medium_isect.medium_id >= 0) {
            // We are inside a medium. Sample a distance and compute transmittance.
            // Also update the intersection point.
            auto transmittance = sample(
                scene.mediums[medium_isect.medium_id],
                incoming_ray,
                surface_isect,
                medium_sample,
                &medium_isect,
                &medium_point);
            throughputs[pixel_id] *= transmittance;
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Intersection *surface_isects;
    const Ray *incoming_rays;
    const MediumSample *medium_samples;
    Intersection *medium_isects;
    Vector3 *medium_points;
    Vector3 *throughputs;
};

void sample_medium(const Scene &scene,
                   const BufferView<int> &active_pixels,
                   const BufferView<Intersection> &surface_isects,
                   const BufferView<Ray> &incoming_rays,
                   const BufferView<MediumSample> &medium_samples,
                   BufferView<Intersection> medium_isects,
                   BufferView<Vector3> medium_points,
                   BufferView<Vector3> throughputs) {
    parallel_for(medium_sampler{
        get_flatten_scene(scene),
        active_pixels.begin(),
        surface_isects.begin(),
        incoming_rays.begin(),
        medium_samples.begin(),
        medium_isects.begin(),
        medium_points.begin(),
        throughputs.begin()},
        active_pixels.size(), scene.use_gpu);
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
    if (medium.type == MediumType::homogeneous) {
        auto h = medium.homogeneous;
        // Sample a distance within the medium and compare it
        // to tmax to see if we are inside of a medium or not
        auto dist = sample_distance(medium, sample);
        auto t = min(dist, ray.tmax);
        auto inside_medium = t < ray.tmax;

        // Compute the transmittance and sampling density
        auto tr = exp(-h.sigma_t * min(t, MaxFloat));

        // Return the weighting factor for scattering inside of a homogeneous medium
        // and its derivatives
        auto density = inside_medium ? (h.sigma_t * tr) : tr;
        auto pdf = sum(density) / 3;

        // Update intersection data
        *medium_point = ray.org + ray.dir * t;
        if (inside_medium) {
            // Set the triangle ID to -1 if we are inside a medium
            medium_isect->tri_id = -1;
        } else {
            // Left our current medium. Update intersection data accordingly
            if (medium_isect->prev_medium_id >= 0) {
                // We previously encountered a medium
                medium_isect->medium_id = medium_isect->prev_medium_id;
                medium_isect->shape_id = medium_isect->prev_shape_id;
                medium_isect->prev_medium_id = -1;
                medium_isect->prev_shape_id = -1;
                medium_isect->tri_id = -1;
            } else {
                // No prior intersection was made. Take the surface intersection
                assert(surface_isect.valid());
                *medium_isect = surface_isect;
            }
        }

        // Calculate the partial derivative of the transmittance with respect to t
        // We assume that f'/pdf is equivalent to (f/pdf)' in terms of expectation
        // auto beta = inside_medium ? (tr * h.sigma_s / pdf) : (tr / pdf);
        auto d_beta_tr = inside_medium ? (tr * h.sigma_s / pdf) : (tr / pdf);

        if (inside_medium) {
            // Only make this add if we are inside a medium. The derivative
            // of beta with respect to sigma_s should not change otherwise
            // auto beta = tr * sigma_s / pdf;
            auto d_beta_sigma_s = tr * d_beta_tr / pdf;
            atomic_add(d_medium.sigma_s[0], d_beta_sigma_s[0]);
            atomic_add(d_medium.sigma_s[1], d_beta_sigma_s[1]);
            atomic_add(d_medium.sigma_s[2], d_beta_sigma_s[2]);
        }

        // TODO Check if we need to update medium_point
        //*medium_point = ray.org + ray.dir * t;

        // We assume min(t, MaxFloat) will always be less than MaxFloat
        // for simplification purposes
        // auto tr = exp(-h.sigma_t * min(t, MaxFloat))
        auto d_tr_t = h.sigma_t * -exp(-h.sigma_t * d_beta_tr);
        auto d_tr_sigma_t = d_beta_tr * -exp(-h.sigma_t * d_beta_tr);

        // Need to recover d_sigma_a and d_sigma_s from d_tr_sigma_t since
        // sigma_t = sigma_a + sigma_s
        auto d_sigma_a = h.sigma_s + d_tr_sigma_t;
        atomic_add(d_medium.sigma_a[0], d_sigma_a[0]);
        atomic_add(d_medium.sigma_a[1], d_sigma_a[1]);
        atomic_add(d_medium.sigma_a[2], d_sigma_a[2]);

        auto d_sigma_s = h.sigma_a + d_tr_sigma_t;
        atomic_add(d_medium.sigma_s[0], d_sigma_s[0]);
        atomic_add(d_medium.sigma_s[1], d_sigma_s[1]);
        atomic_add(d_medium.sigma_s[2], d_sigma_s[2]);

        // auto t = min(dist, ray.tmax)
        // This can introduce a discontinuity!
        auto d_t = min(d_tr_t, ray.tmax);

        // auto dist = sample_distance(medium, sample);
        auto d_dist = Real(0);
        d_sample_distance(medium, sample, d_dist, d_medium);

        // Backpropagate the derivative of the transmittance with respect
        // to t_max to the intersection function.
        auto ray_diff = RayDifferential {
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}
        };
        auto d_v0 = Vector3{0, 0, 0};
        auto d_v1 = Vector3{0, 0, 0};
        auto d_v2 = Vector3{0, 0, 0};
        auto d_ray_diff = RayDifferential {
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}
        };
        d_intersect(Vector3{0, 0, 0},
                    Vector3{0, 0, 0},
                    Vector3{0, 0, 0},
                    ray,
                    ray_diff,
                    d_t,
                    Vector2{0, 0},
                    Vector2{0, 0},
                    Vector2{0, 0},
                    d_v0,
                    d_v1,
                    d_v2,
                    d_ray,
                    d_ray_diff);
    }
}

struct d_medium_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &surface_isect = surface_isects[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        const auto &medium_sample = medium_samples[pixel_id];
        auto &medium_isect = medium_isects[pixel_id];
        auto &medium_point = medium_points[pixel_id];
        auto &d_medium = d_mediums[medium_isect.medium_id];

        if (medium_isect.medium_id >= 0) {
            // TODO Check if d_output = d_throughput
            // throughputs[pixel_id] *= transmittance;
            auto d_transmittance = transmittances[pixel_id] * d_throughputs[pixel_id];
            auto d_ray = d_rays[pixel_id];
            d_sample(scene.mediums[medium_isect.medium_id],
                     incoming_ray,
                     surface_isect,
                     medium_sample,
                     d_transmittance,
                     d_medium,
                     &medium_isect,
                     &medium_point,
                     d_ray);
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Intersection *surface_isects;
    const Ray *incoming_rays;
    const MediumSample *medium_samples;
    const Vector3 *transmittances;
    const Vector3 *d_throughputs;
    DMedium *d_mediums;
    DRay *d_rays;
    Intersection *medium_isects;
    Vector3 *medium_points;
    Vector3 *throughputs;
};

void d_sample_medium(const Scene &scene,
                     const BufferView<int> &active_pixels,
                     const BufferView<Intersection> &surface_isects,
                     const BufferView<Ray> &incoming_rays,
                     const BufferView<MediumSample> &medium_samples,
                     const BufferView<Vector3> &transmittances,
                     const BufferView<Vector3> &d_throughputs,
                     DScene *d_scene,
                     BufferView<DRay> &d_rays,
                     BufferView<Intersection> medium_isects,
                     BufferView<Vector3> medium_points,
                     BufferView<Vector3> throughputs) {
    parallel_for(d_medium_sampler {
        get_flatten_scene(scene),
        active_pixels.begin(),
        surface_isects.begin(),
        incoming_rays.begin(),
        medium_samples.begin(),
        transmittances.begin(),
        d_throughputs.begin(),
        d_scene->mediums.data,
        d_rays.begin(),
        medium_isects.begin(),
        medium_points.begin(),
        throughputs.begin()},
        active_pixels.size(), scene.use_gpu);
}

DEVICE
inline
Vector3 transmittance(const Medium &medium,
                      const Ray &ray) {
    if (medium.type == MediumType::homogeneous) {
        auto h = medium.homogeneous;
        // Use Beer's Law to calculate transmittance
        return exp(-h.sigma_t * min(ray.tmax, MaxFloat));
    } else {
        return Vector3{0, 0, 0};
    }
}

struct transmittance_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &ray = rays[pixel_id];
        transmittances[pixel_id] = Vector3{1, 1, 1};
        // tmax <= 0 means the ray is occluded
        if (medium_isects[pixel_id].medium_id >= 0 && ray.tmax > 0) {
            transmittances[pixel_id] = transmittance(
                mediums[medium_isects[pixel_id].medium_id], ray);
        }
    }

    const Medium *mediums;
    const int *active_pixels;
    const Ray *rays;
    const Intersection *medium_isects;
    Vector3 *transmittances;
};

void evaluate_transmittance(const Scene &scene,
                            const BufferView<int> &active_pixels,
                            const BufferView<Ray> &rays,
                            const BufferView<Intersection> &medium_isects,
                            BufferView<Vector3> transmittances) {
    // TODO: Currently we assume the first surface/volume we hit has a different
    // index of refraction or is fully opaque. This can be inefficient for some
    // cases, e.g. if we have an object with IOR=1 blocking the light.
    // To resolve this we want to repeatedly intersect with the scene until
    // we reach the end of the rays or we hit a surface with different IOR or is opaque.
    parallel_for(transmittance_sampler{
        scene.mediums.data,
        active_pixels.begin(),
        rays.begin(),
        medium_isects.begin(),
        transmittances.begin()},
        active_pixels.size(), scene.use_gpu);
}

/**
 * Backpropagate the derivative of the transmittance with respect to t_max.
 */
DEVICE
inline
void d_transmittance(const Medium &medium,
                     const Ray &ray,
                     const Vector3 &d_output,
                     DRay &d_ray,
                     DMedium &d_medium) {
    if (medium.type == MediumType::homogeneous) {
        auto h = medium.homogeneous;

        // We assume that tmax will always be less than or equal to MaxFloat
        // so that the derivative will not be discontinuous if tmax > MaxFloat
        auto tr = exp(-h.sigma_t * min(ray.tmax, MaxFloat));

        // Backpropagate
        // TODO Figure out if we need to add d_output into this
        auto d_tr_t = h.sigma_t * -exp(-h.sigma_t * min(ray.tmax, MaxFloat));
        auto d_tr_sigma_t = h.sigma_t * -exp(-h.sigma_t * d_tr_t);

        // Need to recover d_sigma_a and d_sigma_s from d_tr_sigma_t since
        // sigma_t = sigma_a + sigma_s
        auto d_sigma_a = h.sigma_s + d_tr_sigma_t;
        atomic_add(&d_medium.sigma_a[0], d_sigma_a[0]);
        atomic_add(&d_medium.sigma_a[1], d_sigma_a[1]);
        atomic_add(&d_medium.sigma_a[2], d_sigma_a[2]);

        auto d_sigma_s = h.sigma_a + d_tr_sigma_t;
        atomic_add(&d_medium.sigma_s[0], d_sigma_s[0]);
        atomic_add(&d_medium.sigma_s[1], d_sigma_s[1]);
        atomic_add(&d_medium.sigma_s[2], d_sigma_s[2]);

        // Backpropagate the derivative of the transmittance with respect
        // to t_max to the intersection function.
        auto ray_diff = RayDifferential {
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}
        };
        auto d_v0 = Vector3{0, 0, 0};
        auto d_v1 = Vector3{0, 0, 0};
        auto d_v2 = Vector3{0, 0, 0};
        auto d_ray_diff = RayDifferential {
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}
        };
        d_intersect(Vector3{0, 0, 0},
                    Vector3{0, 0, 0},
                    Vector3{0, 0, 0},
                    ray,
                    ray_diff,
                    d_tr_t,
                    Vector2{0, 0},
                    Vector2{0, 0},
                    Vector2{0, 0},
                    d_v0,
                    d_v1,
                    d_v2,
                    d_ray,
                    d_ray_diff);
    }
}

struct d_transmittance_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &ray = rays[pixel_id];
        auto &d_medium = d_mediums[pixel_id];

        // tmax <= 0 means the ray is occluded
        if (medium_isects[pixel_id].medium_id >= 0 && ray.tmax > 0) {
            // TODO Check if this how we get d_tr or if we just
            // initialize a Vector3{1, 1, 1}
            auto &d_tr = transmittances[pixel_id];
            auto d_ray = d_rays[pixel_id];
            d_transmittance(mediums[medium_isects[pixel_id].medium_id],
                            ray, d_tr, d_ray, d_medium);
        }
    }

    const Medium *mediums;
    const int *active_pixels;
    const Ray *rays;
    const Intersection *medium_isects;
    const Vector3 *transmittances;
    DMedium *d_mediums;
    DRay *d_rays;
};

void d_evaluate_transmittance(const Scene &scene,
                              const BufferView<int> &active_pixels,
                              const BufferView<Ray> &rays,
                              const BufferView<Intersection> &medium_isects,
                              const BufferView<Vector3> &transmittances,
                              BufferView<DMedium> d_mediums,
                              BufferView<DRay> d_rays) {
    parallel_for(d_transmittance_sampler{
        scene.mediums.data,
        active_pixels.begin(),
        rays.begin(),
        medium_isects.begin(),
        transmittances.begin(),
        d_mediums.begin(),
        d_rays.begin()},
        active_pixels.size(), scene.use_gpu);
}