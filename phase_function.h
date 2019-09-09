#pragma once

#include "vector.h"

// Defining the inverse for 1/4PI here
static const float INV_4PI = 0.07957747154594766788;

/*
 * An interface to store various information for a phase function.
 * Useful when sampling participating media. Wi and Wo are assumed
 * to be pointing outward, as is the convention in the PBRT book
 */
struct PhaseFunction {
    virtual double p(const Vector3 &wo, const Vector3 &wi) const = 0;
    virtual double sample_p(const Vector3 &wo, Vector3 *wi,
                            const Vector2 &sample) const = 0;
};

// An implementation of the Heyney-Greenstein phase function
struct HenyeyGreenstein : PhaseFunction {
   public:
    HenyeyGreenstein(float g) : g(g){};

    double p(const Vector3 &wo, const Vector3 &wi) const override {
        return PhaseHG(dot(wo, wi), g);
    }

    // This function works much like p() but it is extended
    // to use a sample in the range of [0, 1)^2 to perform MIS
    double sample_p(const Vector3 &wo, Vector3 *wi,
                    const Vector2 &sample) const override {
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
        float phi = 2.0f * M_PI * sample[1];
        Vector3 v1, v2;
        coordinate_system(wo, v1, v2);
        *wi = sphericalDirection(sinTheta, cosTheta, phi, v1, v2, -wo);
        return PhaseHG(-cosTheta, g);
    }

   private:
    // Calculate the phase based on the angle between wo and wi
    // and a scattering factor g
    inline float PhaseHG(float cosTheta, float g) const {
        float denom = 1.0f + g * g + 2.0f * g * cosTheta;
        return INV_4PI * (1.0f - g * g) / (denom * std::sqrt(denom));
    }

    // Calculates the new direction of the outgoing vector given three basis
    // vectors
    inline Vector3 sphericalDirection(float sinTheta, float cosTheta, float phi,
                                      const Vector3 &x, const Vector3 &y,
                                      const Vector3 &z) const {
        return sinTheta * std::cos(phi) * x + sinTheta * std::sin(phi) * y +
               cosTheta * z;
    }

    const float g;
};