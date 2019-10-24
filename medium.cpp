#include "medium.h"
#include "medium_interaction.h"

HomogeneousMedium::HomogeneousMedium(const Vector3 &sigma_a,
                                     const Vector3 &sigma_s, float g)
    : sigma_a(sigma_a), sigma_s(sigma_s), sigma_t(sigma_a + sigma_s), g(g){};

Vector3 HomogeneousMedium::transmittance(const Ray &ray,
                                         const MediumSample &sample) const {
    // Use Beer's Law to calculate transmittance
    return vecExp(-sigma_t * min(static_cast<float>(ray.tmax * length(ray.dir)),
                                 MAXFLOAT));
}

/**
 * Sample the given medium to see if the ray is affected by it. The
 * transmittance is automatically calculated and encoded in the returned
 * Vector3
 */
Vector3 HomogeneousMedium::sample(const Ray &ray,
                                  const SurfacePoint &surface_point,
                                  const MediumSample &sample,
                                  ptr<MediumInteraction> mi) const {
    // Sample a channel and distance along the ray
    int channel =
        min(sample.uv[0] * NUM_SAMPLES, static_cast<double>(NUM_SAMPLES - 1));
    double dist = -log(1.0 - sample.uv[1]) / sigma_t[channel];

    float t = min(dist / length(ray.dir), ray.tmax);
    bool sampled_medium = t < ray.tmax;
    if (sampled_medium) {
        // If we are inside the medium we need to sample the
        // phase function. We use HG in this case
        *mi = MediumInteraction(surface_point, -ray.dir, this,
                                new HenyeyGreenstein(g));
    }

    // Compute the transmittance and sampling density
    Vector3 tr = vecExp(-sigma_t * min(t, MAXFLOAT) * length(ray.dir));

    // Return the weighting factor for scattering inside of a homogeneous medium
    Vector3 density = sampled_medium ? (sigma_t * tr) : tr;
    float pdf = 0.0;
    for (uint i = 0; i < NUM_SAMPLES; ++i) {
        pdf += density[i];
    }

    pdf *= 1.0 / NUM_SAMPLES;

    return sampled_medium ? (tr * sigma_s / pdf) : (tr / pdf);
}

// A helper function to calculate e^x component-wise
Vector3 HomogeneousMedium::vecExp(const Vector3 &vec) const {
    Vector3 ret(0.0, 0.0, 0.0);
    for (int i = 0; i < 3; ++i) {
        ret[i] = exp(vec[i]);
    }
    return ret;
}