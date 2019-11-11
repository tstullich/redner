#pragma once

#include "redner.h"
#include "intersection.h"
#include "buffer.h"
#include "phase_function.h"
#include "ptr.h"
#include "py_utils.h"
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
    homogeneous
};

struct HomogeneousMedium {
    HomogeneousMedium(const Vector3f &sigma_a,
                      const Vector3f &sigma_s,
                      float g)
        : sigma_a(sigma_a), sigma_s(sigma_s), sigma_t(sigma_a + sigma_s), g(g) {}

    Vector3f sigma_a, sigma_s, sigma_t;
    float g;
};

struct Medium {
    Medium(const HomogeneousMedium &homogeneous) {
        type = MediumType::homogeneous;
        this->homogeneous = homogeneous;
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

// Evaluate the transmittance between two points.
void evaluate_transmittance(const Scene &scene,
                            const BufferView<int> &active_pixels,
                            const BufferView<Ray> &rays,
                            const BufferView<Intersection> &medium_isects,
                            const BufferView<MediumSample> &medium_samples,
                            BufferView<Vector3> transmittances);
