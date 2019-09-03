#pragma once

#include "medium_interaction.h"
#include "ray.h"
#include "sampler.h"
#include "vector.h"

/*
 * An interface struct to describe a type of medium. The different types
 * can be either homogeneous or heterogeneous
 */
struct Medium {
    virtual ~Medium();
    virtual Vector3f transmittance(const Ray &ray, Sampler &sampler) const = 0;
    virtual Vector3f sample(const Ray &ray, Sampler &sampler,
                            MediumInteraction *mi) const = 0;
};

/**
 * This struct holds data to represent a medium
 */
struct HomogeneousMedium : Medium {
   public:
    HomogeneousMedium(const Vector3f &sigma_a, const Vector3f &sigma_s, float g)
        : sigma_a(sigma_a),
          sigma_s(sigma_s),
          sigma_t(sigma_a + sigma_s),
          g(g){};

    Vector3f transmittance(const Ray &ray, Sampler &sampler) const {
        // Use Beer's Law to calculate transmittance
        return vecExp(
            -sigma_t *
            std::min(static_cast<float>(ray.tmax * length(ray.dir)), MAXFLOAT));
    }

    Vector3f sample(const Ray &ray, Sampler &sampler,
                    MediumInteraction *mi) const {
        // Sample a channel and distance along the ray
        // TODO actually implement this with the sampler
        // int channel = 0;
        float dist = 1.0;  // Dummy for testing

        float t = std::min(dist * length(ray.dir), ray.tmax);
        bool sampledMedium = t < ray.tmax;
        if (sampledMedium) {
            // If we are inside the medium we need to sample the
            // phase function. We use HG in this case
            // TODO Figure out how to get the surface point
            // mi = MediumInteraction(, -ray.dir, this, HeyneyGreenstein(g));
        }

        // Compute the transmittance and sampling density
        Vector3f tr = transmittance(ray, sampler);

        // Return the weighting factor scattering inside of a homogeneous medium
        Vector3f density = sampledMedium ? (sigma_t * tr) : tr;
        float pdf = 0.0f;
        for (int i = 0; i < 3; i++) {
            pdf += density[i];
        }

        pdf *= 1.0f / 3;  // Dummy for testing

        return sampledMedium ? (tr * sigma_s / pdf) : (tr / pdf);
    }

   private:
    // A helper function to calculate e^x component-wise
    Vector3f vecExp(const Vector3f &vec) const {
        Vector3f ret;
        // TODO Check how to implement numSamples from PBRT
        for (int i = 0; i < 3; i++) {
            ret[i] = std::exp(vec[i]);
        }
        return ret;
    }

    const Vector3f sigma_a, sigma_s, sigma_t;
    const float g;
};