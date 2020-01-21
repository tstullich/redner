#pragma once

#include "redner.h"
#include "intersection.h"
#include "buffer.h"
#include "phase_function.h"
#include "ptr.h"
#include "ray.h"
#include "vector.h"

// Forward declarations
struct DScene;
struct MediumInteraction;
struct Sampler;
struct Scene;

template <typename T>
struct TMediumSample {
    TVector2<T> uv;
};

using MediumSample = TMediumSample<Real>;

enum class MediumType {
    homogeneous,
    heterogeneous,
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

/**
 * Class to hold information about a heterogeneous medium.
 */
struct HeterogeneousMedium {
    HeterogeneousMedium(const Vector3f &sigma_a,
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

    Medium(const HeterogeneousMedium &medium) {
        type = MediumType::heterogeneous;
        heterogeneous = medium;
    }

    MediumType type;
    union {
        HomogeneousMedium homogeneous;
        HeterogeneousMedium heterogeneous;
    };
};

DEVICE
inline
PhaseFunction get_phase_function(const Medium &medium) {
    if (medium.type == MediumType::homogeneous) {
        return PhaseFunction(HenyeyGreenstein{medium.homogeneous.g});
    } else if (medium.type == MediumType::heterogeneous) {
        return PhaseFunction(HenyeyGreenstein{medium.heterogeneous.g});
    } else {
        return PhaseFunction();
    }
}

// Function to sample a distance within a medium
Real sample_distance(const Medium &medium, const MediumSample &sample);

// Sample a distance inside the medium and compute the transmittance.
// Update intersection data as well.
void sample_medium(const Scene &scene,
                   const BufferView<int> &active_pixels,
                   const BufferView<Intersection> &surface_isects,
                   const BufferView<Ray> &incoming_rays,
                   const BufferView<MediumSample> &medium_samples,
                   BufferView<Intersection> medium_isects,
                   BufferView<Vector3> medium_points,
                   BufferView<Vector3> throughputs);

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
                     BufferView<Vector3> throughputs);

// Evaluate the transmittance between two points.
void evaluate_transmittance(const Scene &scene,
                            const BufferView<int> &active_pixels,
                            const BufferView<Ray> &rays,
                            const BufferView<Intersection> &medium_isects,
                            BufferView<Vector3> transmittances);

// Backpropagate the transmittance between two points.
void d_evaluate_transmittance(const Scene &scene,
                              const BufferView<int> &active_pixels,
                              const BufferView<Ray> &rays,
                              const BufferView<Intersection> &medium_isects,
                              const BufferView<Vector3> &medium_points,
                              BufferView<Vector3> transmittances,
                              BufferView<Vector3> d_rays);

// Calculate the transmittance of a ray segment given a ray
Vector3 transmittance(const Medium &medium, const Ray &ray);

// Calculate the derivative of the transmittance w.r.t tmax
void d_transmittance(const Medium &medium,
                     const Ray &ray,
                     const Vector3 &d_output,
                     DRay &d_ray,
                     DMedium &d_medium);