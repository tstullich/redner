#pragma once

#include "medium.h"
#include "redner.h"
#include "sampler.h"
#include "vector.h"

struct HomogeneousMedium : public Medium {
public:
  HomogeneousMedium(const float sigma_a, const float sigma_s, float g)
      : sigma_a(sigma_a), sigma_s(sigma_s), sigma_t(sigma_a + sigma_s), g(g){};

  Real transmittance(const Ray &ray, Sampler &sampler);

  Real sample(const Ray &ray, Sampler &sampler,
              MediumInteraction mediumInteraction);

private:
  const Real sigma_a, sigma_s, sigma_t, g;
};