#pragma once

#include "intersection.h"
#include "phase_function.h"
#include "ptr.h"
#include "py_utils.h"
#include "ray.h"
#include "redner.h"
#include "vector.h"

// Forward declarations
struct MediumInteraction;
struct Sampler;

template <typename T>
struct TMediumSample {
    TVector2<T> uv;
};

using MediumSample = TMediumSample<Real>;

/*
 * An interface struct to describe a type of medium. The different types
 * can be either homogeneous or heterogeneous
 */
struct Medium {
    virtual ~Medium();

    DEVICE
    virtual Vector3 transmittance(const Ray &ray,
                                  const MediumSample &sample) const = 0;

    DEVICE
    virtual Vector3 sample(const Ray &ray, const SurfacePoint &surface_point,
                           const MediumSample &sample,
                           ptr<MediumInteraction> mi) const = 0;
};

/*
 * This class is supposed to provide an interface such that the virtual
 * interface functions can be picked up by pybind. If this class is not
 * provided we are not able to provide a proper base class for the
 * different abstractions of the Medium class. Technically these functions
 * should not be directly called within Python but it's necessary we
 * provide bindings for them, otherwise pybind cannot create an instance
 * of the Medium class.
 */
struct PyMedium : Medium {
   public:
    using Medium::Medium;

    DEVICE
    Vector3 transmittance(const Ray &ray,
                          const MediumSample &sample) const override {
        PYBIND11_OVERLOAD_PURE(Vector3, Medium, transmittance, ray, sample);
    }

    DEVICE
    Vector3 sample(const Ray &ray, const SurfacePoint &surface_point,
                   const MediumSample &sample,
                   ptr<MediumInteraction> mi) const override {
        PYBIND11_OVERLOAD_PURE(Vector3, Medium, sample, ray, surface_point,
                               sample, mi);
    }
};

/*
 * This struct holds data to represent a homogeneous medium.
 * Since the medium is assumed to be homogeneous, we use Beer's law in
 * order to calculate the transmittance. MIS is also implemented in
 * the sample() function
 */
struct HomogeneousMedium : Medium {
   public:
    HomogeneousMedium(const Vector3 &sigma_a, const Vector3 &sigma_s, float g);

    DEVICE
    Vector3 transmittance(const Ray &ray,
                          const MediumSample &sample) const override;

    DEVICE
    Vector3 sample(const Ray &ray, const SurfacePoint &surface_point,
                   const MediumSample &sample,
                   ptr<MediumInteraction> mi) const override;

   private:
    // A helper function to calculate e^x component-wise
    Vector3 vecExp(const Vector3 &vec) const;

    const Vector3 sigma_a, sigma_s, sigma_t;
    const float g;
    // Need to change this if we sample more dimensions
    static const uint NUM_SAMPLES = 2;
};