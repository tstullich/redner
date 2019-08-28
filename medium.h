#pragma once

#include "ray.h"
#include "sampler.h"

// Forward declaration for Medium
struct MediumInteraction;

struct Medium {
public:
  virtual ~Medium();
  virtual Real transmittance(const Ray &ray, Sampler &sampler) = 0;
  // TODO Check if we need things like MemoryArena and MediumInteraction
  virtual Real sample(const Ray &ray, Sampler &sampler,
                      MediumInteraction *mediumInteraction) = 0;
};

struct PhaseFunction {
  // TODO Implement
};

struct MediumInteraction {
  // See if we need to abstract from Interaction class
  MediumInteraction(const TVector3<Real> &point, const TVector3<Real> &wo,
                    Real time, const Medium *medium,
                    const PhaseFunction *phase) {}

  bool isValid() const {
    // If interaction in a medium has occured the phase function should have
    // been initialized
    return phase != nullptr;
  }

  const PhaseFunction *phase;
};