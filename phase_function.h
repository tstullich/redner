#pragma once

#include "vector.h"

// Defining the inverse for 1/4PI here
static const float INV_4PI = 0.07957747154594766788;

template <typename T>
struct TPhaseSample {
    TVector2<T> uv;
};

using PhaseSample = TPhaseSample<Real>;

/*
 * An interface to store various information for a phase function.
 * Useful when sampling participating media. Wi and Wo are assumed
 * to be pointing outward, as is the convention in the PBRT book
 */
struct PhaseFunction {
    DEVICE virtual double p(const Vector3 &wo, const Vector3 &wi) const = 0;

    DEVICE virtual Vector3 sample_p(const Vector3 &wo,
                                    const PhaseSample &sample) const = 0;
};

// An implementation of the Henyey-Greenstein phase function
struct HenyeyGreenstein : PhaseFunction {
   public:
    HenyeyGreenstein(float g) : g(g){};

    DEVICE double p(const Vector3 &wo, const Vector3 &wi) const override {
        return PhaseHG(dot(wo, wi), g);
    }

    // This function works much like p() but it is extended
    // to use a sample in the range of [0, 1)^2 to perform MIS
    // Returns the incident vector wi as a result
    DEVICE Vector3 sample_p(const Vector3 &wo,
                            const PhaseSample &sample) const override {
        // Compute cosine theta
        double cos_theta;
        if (abs(g) < 1e-3) {
            cos_theta = 1.0 - 2.0 * sample.uv[0];
        } else {
            double sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * sample.uv[0]);
            cos_theta = (1.0 + g * g - sqr_term * sqr_term) / (2.0 * g);
        }

        // Compute the direction wi based on the HG phase function
        double sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
        double phi = 2.0 * M_PI * sample.uv[1];
        Vector3 v1, v2;
        coordinate_system(wo, v1, v2);
        return sphericalDirection(sin_theta, cos_theta, phi, v1, v2, -wo);
    }

   private:
    // Calculate the phase based on the angle between wo and wi
    // and a scattering factor g
    DEVICE inline double PhaseHG(double cos_theta, float g) const {
        double denom = 1.0 + g * g + 2.0 * g * cos_theta;
        return INV_4PI * (1.0 - g * g) / (denom * sqrt(denom));
    }

    // Calculates the new direction of the outgoing vector given three basis
    // vectors
    DEVICE inline Vector3 sphericalDirection(double sin_theta, double cos_theta,
                                             double phi, const Vector3 &x,
                                             const Vector3 &y,
                                             const Vector3 &z) const {
        return sin_theta * cos(phi) * x + sin_theta * sin(phi) * y +
               cos_theta * z;
    }

    const float g;
};