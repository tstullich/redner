#include "homogeneous_medium.h"

// Calculate the transmittance of a light ray passing through the homogeneous
// medium
Real HomogeneousMedium::transmittance(const Ray &ray, Sampler &sampler) {
  // Use Beer's Law to evaluate transmittance
  return std::exp(-sigma_t * std::min(ray.tmax * length(ray.dir),
                                      std::numeric_limits<double>::max()));
}

Real HomogeneousMedium::sample(const Ray &ray, Sampler &sampler,
                               MediumInteraction mediumInteraction) {
  // Sample a random channel and distance along ray
  // int channel = min()
  Real dist = 1.0; // Implement
  Real t = std::min(dist * length(ray.dir), ray.tmax);

  // Calculate transmittance and sampling density using Beer's Law
  // TODO Real should actually be an n-dimensional array instead of scalar
  Real tr =
      std::exp(-sigma_t * std::min(t, std::numeric_limits<double>::max()) *
               length(ray.dir));

  bool sampledMedium = t < ray.tmax;
  if (sampledMedium) {
    // A medium interaction has occured and so we add the needed information
    // mediumInteraction =
  }

  // Calculate the weighting factor for scattering within the medium
  Real density = sampledMedium ? (sigma_t * tr) : tr;
  int nSamples = 4; // TODO Remove this with actual sample number
  Real pdf = 0;
  for (int i = 0; i < nSamples; ++i) {
    pdf += density;
  }
  // Normalize PDF
  pdf *= 1.0 / nSamples;

  return sampledMedium ? (tr * sigma_s / pdf) : (tr / pdf);
}