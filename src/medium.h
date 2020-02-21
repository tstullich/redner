#pragma once

#include "redner.h"
#include "atomic.h"
#include "buffer.h"
#include "intersection.h"
#include "phase_function.h"
#include "ptr.h"
#include "ray.h"
#include "thrust_utils.h"
#include "vector.h"

// Forward declarations
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

struct TransmittanceBuffer {
    TransmittanceBuffer() {}

    TransmittanceBuffer(int num_pixels, bool use_gpu) {
        active_pixels = Buffer<int>(use_gpu, num_pixels);
        rays = Buffer<Ray>(use_gpu, num_pixels);
        surface_isects = Buffer<Intersection>(use_gpu, num_pixels);
        surface_points = Buffer<SurfacePoint>(use_gpu, num_pixels);
        medium_ids = Buffer<int>(use_gpu, num_pixels);
    }

    int num_pixels;
    Buffer<int> active_pixels;
    Buffer<Ray> rays;
    Buffer<Intersection> surface_isects;
    Buffer<SurfacePoint> surface_points;
    Buffer<int> medium_ids;
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

// Function to sample a distance within a medium
Real sample_distance(const Medium &medium, const MediumSample &sample);

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
void trace_transmittance_rays(const Scene &scene,
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

// Given rays, intersect with scene until
// hitting an opaque object. Return the first
// and the last intersection. Evaluate transmittance
// along the way.
void intersect_and_eval_transmittance(const Scene &scene,
                                      const BufferView<int> &active_pixels,
                                      const BufferView<int> &medium_ids,
                                      const BufferView<Ray> &outgoing_rays,
                                      const BufferView<RayDifferential> &ray_differentials,
                                      BufferView<Intersection> first_intersections,
                                      BufferView<SurfacePoint> first_points,
                                      BufferView<RayDifferential> new_ray_differentials,
                                      BufferView<Intersection> last_intersections,
                                      BufferView<SurfacePoint> last_points,
                                      BufferView<Vector3> transmittances,
                                      TransmittanceBuffer &tr_buffer,
                                      ThrustCachedAllocator &thrust_alloc,
                                      BufferView<OptiXRay> optix_rays,
                                      BufferView<OptiXHit> optix_hits);

void d_intersect_and_eval_transmittance(const Scene &scene,
                                        const BufferView<int> &active_pixels,
                                        const BufferView<int> &medium_ids,
                                        const BufferView<Ray> &outgoing_rays,
                                        const BufferView<RayDifferential> &ray_differentials,
                                        BufferView<Intersection> first_intersections,
                                        BufferView<SurfacePoint> first_points,
                                        BufferView<RayDifferential> new_ray_differentials,
                                        BufferView<Intersection> last_intersections,
                                        BufferView<SurfacePoint> last_points,
                                        BufferView<Vector3> transmittances,
                                        TransmittanceBuffer &tr_buffer,
                                        ThrustCachedAllocator &thrust_alloc,
                                        BufferView<OptiXRay> optix_rays,
                                        BufferView<OptiXHit> optix_hits);

// // Backpropagate the transmittance between two points.
// void d_evaluate_transmittance(const Scene &scene,
//                               const BufferView<int> &active_pixels,
//                               const BufferView<Ray> &rays,
//                               const BufferView<Intersection> &medium_isects,
//                               const BufferView<Vector3> &d_transmittances,
//                               BufferView<DMedium> d_mediums,
//                               BufferView<DRay> d_rays);
