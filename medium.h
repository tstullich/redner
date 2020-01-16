#pragma once

#include "redner.h"
#include "intersection.h"
#include "buffer.h"
#include "phase_function.h"
#include "ray.h"
#include "vector.h"

// Forward declarations
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

    DEVICE inline MediumType get_type() const {
        return type;
    }

    DEVICE inline Vector3f get_sigma_a() const {
        return type == MediumType::homogeneous ? homogeneous.sigma_a :
                                                 heterogeneous.sigma_a;
    }

    DEVICE inline Vector3f get_sigma_s() const {
        return type == MediumType::homogeneous ? homogeneous.sigma_s :
                                                 heterogeneous.sigma_s;
    }

    DEVICE inline float get_g() const {
        return type == MediumType::homogeneous ? homogeneous.g :
                                                 heterogeneous.g;
    }

    DEVICE inline void set_sigma_a(const Vector3 &sigma_a) {
        if (type == MediumType::homogeneous) {
            homogeneous.sigma_a = sigma_a;
        } else {
            heterogeneous.sigma_a = sigma_a;
        }
    }

    DEVICE inline void set_sigma_s(const Vector3 &sigma_s) {
        if (type == MediumType::homogeneous) {
            homogeneous.sigma_s = sigma_s;
        } else {
            heterogeneous.sigma_s = sigma_s;
        }
    }

    DEVICE inline void set_g(const float &g) {
        if (type == MediumType::homogeneous) {
            homogeneous.g = g;
        } else {
            heterogeneous.g = g;
        }
    }

    DEVICE inline void set_type(const MediumType &medium_type) {
        type = medium_type;
    }
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
                              BufferView<Vector3> transmittances);

// Calculate the transmittance of a ray segment given a ray
Vector3 transmittance(const Medium &medium, const Ray &ray);

// Calculate the derivative of the transmittance w.r.t tmax
void d_transmittance(const Medium &medium, const Ray &ray, DRay &d_ray);