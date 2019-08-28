#pragma once

#include "ray.h"
#include "sampler.h"

struct MediumInteraction {};

struct Medium {
public:
  virtual ~Medium();
  virtual Real transmittance(const Ray &ray, Sampler &sampler) = 0;
  // TODO Check if we need things like MemoryArena and MediumInteraction
  virtual Real sample(const Ray &ray, Sampler &sampler,
                      MediumInteraction mediumInteraction) = 0;
};