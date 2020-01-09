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

// Calculate the phase function based on the angle between wo and wi
// and a scattering factor g
DEVICE
inline
Real phase_HG(Real cos_theta, float g) {
    auto denom = 1 + g * g + 2 * g * cos_theta;
    return Real(INV_4PI) * (1 - g * g) / (denom * sqrt(denom));
}

// Evaluate the phase function at a point given incoming and outgoing direction.
// The directions are assumed to be pointed outwards.
DEVICE
inline
Real phase_function(const PhaseFunction &phase_function,
                    const Vector3 &wo,
                    const Vector3 &wi) {
    if (phase_function.type == PhaseFunctionType::HenyeyGreenstein) {
        auto hg = phase_function.hg;
        return phase_HG(dot(wo, wi), hg.g);
    }  else {
        return 0;
    }
}

DEVICE
inline
Real d_phase_HG(Real cos_theta, float g) {
    // Take the partial derivative of the phase function with respect to cos_theta
    // auto denom = 1 + g * g + 2 * g * cos_theta;
    auto d_denom = 2 * g;

    // auto phase_HG = Real(INV_4PI) * (1 - g * g) / (denom * sqrt(denom));
    auto d_pHG_cos = (Real(3 * M_PI) * (g * g - 1)) / (8 * pow(d_denom, 2.5));

    // TODO Need to figure out where we need to backpropagate d_pHG_cos

    return d_pHG_cos;
}


// Evaluate the derivative of the phase function. It works much like
// d_pdf_phase() so perhaps we can refactor that a bit.
DEVICE
inline
Real d_phase_function(const PhaseFunction &phase_function,
                      const Vector3 &wo,
                      const Vector3 &wi) {
    if (phase_function.type == PhaseFunctionType::HenyeyGreenstein) {
        auto hg = phase_function.hg;
        return d_phase_HG(dot(wo, wi), hg.g);
    } else {
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

// PDF evaluation.
DEVICE
inline
Real phase_function_pdf(const PhaseFunction &phase_function,
                        const Vector3 &wo,
                        const Vector3 &wi) {
    if (phase_function.type == PhaseFunctionType::HenyeyGreenstein) {
        auto hg = phase_function.hg;
        return phase_HG(dot(wo, wi), hg.g);
    }  else {
        return 0;
    }
}

// Derivative of the PDF evaluation. This returns the partial derivative
// of the Henyey-Greenstein function with respect to cos_theta.
DEVICE
inline
Real d_phase_function_pdf(const PhaseFunction &phase_function,
                          const Vector3 &wo,
                          const Vector3 &wi) {
    if (phase_function.type == PhaseFunctionType::HenyeyGreenstein) {
        auto hg = phase_function.hg;
        return d_phase_HG(dot(wo, wi), hg.g);
    } else {
        return 0;
    }
}