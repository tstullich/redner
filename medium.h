#pragma once

#include "phase_function.h"
#include "ray.h"
#include "vector.h"
#include "intersection.h"

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
    virtual Vector3f transmittance(const Ray &ray,
                                   const MediumSample &sample) const = 0;
    virtual Vector3f sample(const Ray &ray, const SurfacePoint &surface_point,
                            const MediumSample &sample,
                            MediumInteraction *mi) const = 0;
};

/**
 * This struct holds data to represent a homogeneous medium.
 * Since the medium is assumed to be homogeneous, we use Beer's law in
 * order to calculate the transmittance. Sampling is done by
 */
struct HomogeneousMedium : Medium {
   public:
    HomogeneousMedium(const Vector3f &sigma_a, const Vector3f &sigma_s,
                      float g);
    Vector3f transmittance(const Ray &ray, const MediumSample &sample) const;
    Vector3f sample(const Ray &ray, const SurfacePoint &surface_point,
                    const MediumSample &sample, MediumInteraction *mi) const;

   private:
    // A helper function to calculate e^x component-wise
    Vector3f vecExp(const Vector3f &vec) const;

    const Vector3f sigma_a, sigma_s, sigma_t;
    const float g;
};