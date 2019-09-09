#include "medium.h"
#include "medium_interaction.h"

HomogeneousMedium::HomogeneousMedium(const Vector3f &sigma_a,
                                     const Vector3f &sigma_s, float g)
    : sigma_a(sigma_a), sigma_s(sigma_s), sigma_t(sigma_a + sigma_s), g(g){};

Vector3f HomogeneousMedium::transmittance(const Ray &ray,
                                          const Vector2f &sample) const {
    // Use Beer's Law to calculate transmittance
    return vecExp(
        -sigma_t *
        std::min(static_cast<float>(ray.tmax * length(ray.dir)), MAXFLOAT));
}

Vector3f HomogeneousMedium::sample(const Ray &ray,
                                   const SurfacePoint &surface_point,
                                   const Vector2f &sample,
                                   MediumInteraction *mi) const {
    // Sample a channel and distance along the ray
    int channel = std::min(sample[0] * NUM_SAMPLES,
                           static_cast<float>(NUM_SAMPLES - 1));
    float dist = -std::log(1.0f - sample[1]) / sigma_t[channel];

    float t = std::min(dist / length(ray.dir), ray.tmax);
    bool sampledMedium = t < ray.tmax;
    if (sampledMedium) {
        // If we are inside the medium we need to sample the
        // phase function. We use HG in this case
        *mi = MediumInteraction(surface_point, -ray.dir, this,
                                HeyneyGreenstein(g));
    }

    // Compute the transmittance and sampling density
    Vector3f tr = vecExp(-sigma_t * std::min(t, MAXFLOAT) * length(ray.dir));

    // Return the weighting factor scattering inside of a homogeneous medium
    Vector3f density = sampledMedium ? (sigma_t * tr) : tr;
    float pdf = 0.0f;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        pdf += density[i];
    }

    pdf *= 1.0f / NUM_SAMPLES;

    return sampledMedium ? (tr * sigma_s / pdf) : (tr / pdf);
}

// A helper function to calculate e^x component-wise
Vector3f HomogeneousMedium::vecExp(const Vector3f &vec) const {
    Vector3f ret(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 3; i++) {
        ret[i] = std::exp(vec[i]);
    }
    return ret;
}