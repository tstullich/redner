#pragma once

#include "vector.h"

// Defining the inverse for 1/4PI here
static const float INV_PI = 0.07957747154594766788;

/*
 * An interface to store various information for a phase function.
 * Useful when sampling participating media. Wi and Wo are assumed
 * to be pointing outward, as is the convention in the PBRT book
 */
struct PhaseFunction {
    virtual float p(const Vector3f &wo, const Vector3f &wi) const = 0;
    virtual float sample_p(const Vector3f &wo, const Vector3f *wi,
                           const Vector2f &u) const = 0;
};

// An implementation of the Heyney-Greenstein phase function
struct HeyneyGreenstein : PhaseFunction {
   public:
    HeyneyGreenstein(float g) : g(g){};

    float p(const Vector3f &wo, const Vector3f &wi) const {
        return PhaseHG(dot(wo, wi), g);
    }

    // This function works much like p() but it is extended
    // to use a sample in the range of [0, 1)^2 to perform MIS
    float sample_p(const Vector3f &wo, const Vector3f *wi,
                   const Vector2f &sample) const {
        // Compute cosine theta
        float cosTheta;
        if (std::abs(g) < 1e-3) {
            cosTheta = 1.0f - 2.0f * sample[0];
        } else {
            float sqrTerm = (1.0f - g * g) / (1.0f - g + 2.0f * g * sample[0]);
            cosTheta = (1.0f + g * g - sqrTerm * sqrTerm) / (2.0f * g);
        }

        // Compute the direction wi based on the HG phase function
        float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = 2.0 * M_PI * sample[1];
        Vector3f v1, v2;
        // TODO lookup CoordinateSystem function and SphericalDirection

        return PhaseHG(-cosTheta, g);
    }

   private:
    // Calculate the phase based on the angle between wo and wi
    // and a scattering factor g
    inline float PhaseHG(float cosTheta, float g) const {
        float denom = 1.0f + g * g + 2.0f * g * cosTheta;
        return INV_PI * (1.0f - g * g) / (denom * std::sqrt(denom));
    }

    const float g;
};