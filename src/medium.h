#pragma once

#include "redner.h"
#include "atomic.h"
#include "buffer.h"
#include "intersection.h"
#include "phase_function.h"
#include "ptr.h"
#include "ray.h"
#include "shape.h"
#include "thrust_utils.h"
#include "vector.h"

struct DScene;
struct Sampler;
struct Scene;

template <typename T>
struct TMediumSample {
    TVector2<T> uv;
};

using MediumSample = TMediumSample<Real>;

enum class MediumType {
    homogeneous
};

// Struct to hold information about the derivatives of a medium
struct DMedium {
    DMedium(const MediumType &type,
            const ptr<float> sigma_a,
            const ptr<float> sigma_s,
            const ptr<float> g) :
        type(type), sigma_a(sigma_a.get()), sigma_s(sigma_s.get()), g(g.get()) {}

    MediumType type;
    float *sigma_a, *sigma_s;
    float *g;
};

/**
 * Class to hold information about a homogeneous medium.
 */
struct HomogeneousMedium {
    HomogeneousMedium(const Vector3f &sigma_a,
                      const Vector3f &sigma_s,
                      float g)
        : sigma_a(sigma_a), sigma_s(sigma_s), sigma_t(sigma_a + sigma_s), g(g) {}

    Vector3f sigma_a, sigma_s, sigma_t;
    float g;
};

struct Medium {
    Medium(const HomogeneousMedium &medium) {
        type = MediumType::homogeneous;
        homogeneous = medium;
    }

    MediumType type;
    union {
        HomogeneousMedium homogeneous;
    };
};

DEVICE
inline
PhaseFunction get_phase_function(const Medium &medium) {
    if (medium.type == MediumType::homogeneous) {
        return PhaseFunction(HenyeyGreenstein{medium.homogeneous.g});
    } else {
        return PhaseFunction();
    }
}

DEVICE
inline
void d_get_phase_function(const Medium &medium,
                          const DPhaseFunction &d_phase_function,
                          DMedium &d_medium) {
    if (medium.type == MediumType::homogeneous) {
        atomic_add(d_medium.g, d_phase_function.hg.g);
    } else {
        return;
    }
}

/// Calculate the transmittance of a ray segment given a ray
DEVICE
inline
Vector3 transmittance(const Medium &medium, const Ray &ray) {
    if (medium.type == MediumType::homogeneous) {
        // Use Beer's Law to calculate transmittance
        if (ray.tmax >= MaxFloat) {
            return Vector3{0, 0, 0};
        } else {
            return exp(-medium.homogeneous.sigma_t * ray.tmax);
        }
    } else {
        return Vector3{0, 0, 0};
    }
}

DEVICE
inline
void d_transmittance(const Medium &medium,
                     const Ray &ray,
                     const Vector3 &d_output,
                     DMedium &d_medium,
                     Ray &d_ray) {
    if (medium.type == MediumType::homogeneous) {
        // output = exp(-medium.homogeneous.sigma_t * min(distance, MaxFloat));
        if (ray.tmax < MaxFloat) {
            auto output = exp(medium.homogeneous.sigma_t * ray.tmax);
            auto d_sigma_t = -d_output * output * ray.tmax;
            d_ray.tmax += -sum(d_output * output * medium.homogeneous.sigma_t);
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

/**
 * Sample the given medium to see if the ray is affected by it.
 * Returns the transmittance divided by pdf.
 */
DEVICE
inline
Vector3 sample_medium(const Medium *mediums,
                      const Shape *shapes,
                      int medium_id,
                      const Ray &ray,
                      const Intersection &surface_isect,
                      const SurfacePoint &surface_point,
                      const MediumSample &sample,
                      int *next_medium_id,
                      Real *medium_distance) {
    const auto &medium = mediums[medium_id];
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
                const auto &shape = shapes[surface_isect.shape_id];
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

DEVICE
inline
void d_sample_medium(const Medium &medium,
                     const Ray &ray,
                     const Intersection &surface_isect,
                     const MediumSample &sample,
                     const Vector3 &d_output,
                     const Real d_distance,
                     DMedium &d_medium,
                     Real &d_tmax) {
    if (medium.type == MediumType::homogeneous) {
        auto h = medium.homogeneous;
        // Sample a distance within the medium and compare it
        // to tmax to see if we are inside of a medium or not
        auto channel = min(int(sample.uv[0] * 3), 2);
        if (h.sigma_t[channel] <= 0) {
            return;
        }
        auto dist = Real(-log(max(1 - sample.uv[1], Real(1e-20))) / h.sigma_t[channel]);
        auto t = min(dist, ray.tmax);
        if (t >= MaxFloat) {
            // Output is 0
            return;
        }
        auto inside_medium = t < ray.tmax;

        // Compute the transmittance and sampling density
        auto tr = exp(-h.sigma_t * min(t, MaxFloat));

        // Return the weighting factor for scattering inside of a homogeneous medium
        // and its derivatives
        auto density = inside_medium ? (h.sigma_t * tr) : tr;
        auto pdf = sum(density) / 3;
        if (pdf <= 0) {
            // Shouldn't happen but just to be safe
            return;
        }
        // *medium_distance = t;
        // return inside_medium ? (tr * h.sigma_s / pdf) : (tr / pdf)
        auto d_tr = Vector3{0, 0, 0};
        auto d_pdf = Real(0);
        if (inside_medium) {
            d_tr += d_output * h.sigma_s / pdf;
            auto d_sigma_s = d_output * tr / pdf;
            d_pdf += (-sum(d_output * tr * h.sigma_s) / square(pdf));
            atomic_add(&d_medium.sigma_s[0], d_sigma_s[0]);
            atomic_add(&d_medium.sigma_s[1], d_sigma_s[1]);
            atomic_add(&d_medium.sigma_s[2], d_sigma_s[2]);
        } else {
            d_tr += d_output / pdf;
            d_pdf += (-sum(d_output * tr) / square(pdf));
        }
        // pdf = sum(density) / 3
        auto d_density = Vector3{d_pdf, d_pdf, d_pdf} / 3;
        auto d_sigma_t = Vector3{0, 0, 0};
        // density = inside_medium ? (h.sigma_t * tr) : tr
        if (inside_medium) {
            d_sigma_t += d_density * tr;
            d_tr += d_density * h.sigma_t;
        } else {
            d_tr += d_density;
        }
        // tr = exp(-h.sigma_t * min(t, MaxFloat))
        auto d_sigma_t_times_t = d_tr * tr;
        d_sigma_t += (-d_sigma_t_times_t * t); // t < MaxFloat at this point
        auto d_t = -sum(d_sigma_t_times_t * h.sigma_t);
        auto d_dist = d_distance;
        // t = min(dist, ray.tmax)
        if (t < ray.tmax) {
            d_dist += d_t;
        } else {
            d_tmax += d_t;
        }
        // dist = Real(-log(max(1 - sample.uv[1], Real(1e-20))) / h.sigma_t[channel])
        d_sigma_t += (-dist / h.sigma_t[channel]);
        atomic_add(&d_medium.sigma_s[0], d_sigma_t[0]);
        atomic_add(&d_medium.sigma_s[1], d_sigma_t[1]);
        atomic_add(&d_medium.sigma_s[2], d_sigma_t[2]);
        atomic_add(&d_medium.sigma_a[0], d_sigma_t[0]);
        atomic_add(&d_medium.sigma_a[1], d_sigma_t[1]);
        atomic_add(&d_medium.sigma_a[2], d_sigma_t[2]);
    }
}

// Sample a distance inside the medium and compute the transmittance.
// Update intersection data as well.
void sample_medium(const Scene &scene,
                   const BufferView<int> &active_pixels,
                   const BufferView<Intersection> &surface_isects,
                   const BufferView<SurfacePoint> &surface_points,
                   const BufferView<Ray> &incoming_rays,
                   const BufferView<MediumSample> &medium_samples,
                   const BufferView<int> &medium_ids,
                   BufferView<int> next_medium_ids,
                   BufferView<Real> medium_distances,
                   BufferView<Vector3> transmittances);

// Return the intermediate surface int between two points,
// also check whether two two points are blocked by surfaces other than
// the one returned.
// This is for transmittance evaluation: for sampling efficiency,
// we want to skip through all participating media boundaries between
// the two points with IOR = 1.
// However this creates thread divergences and unbounded memory requirement.
// Instead we skip at most *one* boundary with IOR=1.
// The arguments with "int_" are the intermediate information
// we store when skipping through the boundary.
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
                                  BufferView<OptiXHit> optix_hits);

// Given outgoing rays, intersect with at most two surfaces and return them.
// This is for transmittance evaluation: for sampling efficiency,
// we want to skip through all participating media boundaries between
// the two points with IOR = 1.
// However this creates thread divergences and unbounded memory requirement.
// Instead we skip at most *one* boundary with IOR=1.
// The arguments with "int_" are the intermediate information
// we store when skipping through the boundary.
// light_isects & light_points store the second surface.
// If there is only one surface, int_ and light_ store the same surface.
// Also store the distance to light_point in outgoing_rays.tmax
// for edge sampling
void trace_scatter_transmittance_rays(const Scene &scene,
                                      const BufferView<int> &active_pixels,
                                      BufferView<Ray> outgoing_rays,
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
                                      BufferView<OptiXHit> optix_hits);
