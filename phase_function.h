#pragma once

#include "redner.h"
#include "vector.h"
#include "directional_sample.h"

enum class PhaseFunctionType {
    HenyeyGreenstein
};

struct HenyeyGreenstein {
    float g;
};

struct PhaseFunction {
    DEVICE inline
    PhaseFunction() {
        type = PhaseFunctionType::HenyeyGreenstein;
        this->hg.g = 0;
    }

    DEVICE inline
    PhaseFunction(const HenyeyGreenstein &hg) {
        type = PhaseFunctionType::HenyeyGreenstein;
        this->hg = hg;
    }

    PhaseFunctionType type;
    union {
        HenyeyGreenstein hg;
    };
};

// Calculate the phase function probability distribution function
// based on the angle between wo and wi and a scattering factor g
DEVICE
inline
Real phase_HG_pdf(const Vector3 &wi, const Vector3 &wo, const float &g) {
    auto cos_theta = dot(wi, wo);
    auto numerator = Real(INV_4PI) * (1 - g * g);
    auto denominator = 1 + g * g + 2 * g * cos_theta;
    return numerator / (denominator * sqrt(denominator));
}

// Evaluate the phase function at a point given incoming and outgoing direction.
// The directions are assumed to be pointed outwards. This function returns
// the PDF contribution
DEVICE
inline
Real phase_function_pdf(const PhaseFunction &phase_function,
                        const Vector3 &wo,
                        const Vector3 &wi) {
    if (phase_function.type == PhaseFunctionType::HenyeyGreenstein) {
        auto hg = phase_function.hg;
        return phase_HG_pdf(wo, wi, hg.g);
    }  else {
        return 0;
    }
}

// Given an incoming direction, sample an outgoing direction for a phase function.
// The directions are assumed to be pointed outwards.
DEVICE
inline
Vector3 sample_phase_function(const PhaseFunction &phase_function,
                              const Vector3 &wi,
                              const DirectionalSample &sample) {
    if (phase_function.type == PhaseFunctionType::HenyeyGreenstein) {
        auto hg = phase_function.hg;
        auto g = hg.g;
        // Compute cosine theta
        Real cos_theta = 0;
        if (fabs(g) < 1e-3f) {
            cos_theta = 1 - 2 * sample.uv[0];
        } else {
            Real sqr_term = (1 - g * g) / (1 - g + 2 * g * sample.uv[0]);
            cos_theta = (1 + g * g - sqr_term * sqr_term) / (2 * g);
        }

        // Compute the direction wi based on the HG phase function
        Real sin_theta = sqrt(max(Real(0), 1 - cos_theta * cos_theta));
        Real phi = Real(2 * M_PI) * sample.uv[1];
        Vector3 v0, v1;
        coordinate_system(wi, v0, v1);
        return sin_theta * cos(phi) * v0 +
               sin_theta * sin(phi) * v1 +
               cos_theta * -wi;
    } else {
        return Vector3{0, 0, 0};
    }
}