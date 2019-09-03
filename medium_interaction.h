#pragma once

#include "intersection.h"
#include "medium.h"
#include "phase_function.h"

/**
 * This struct tracks how rays should interact when they are at the bounds
 * of a given surface.
 */
struct MediumInteraction {
   public:
    MediumInteraction(const SurfacePoint &p, const Vector3f &wo,
                      const Medium *medium, const PhaseFunction *phase)
        : surfacePoint(p), wo(wo), medium(medium), phase(phase){};

    const Medium *getMedium() const {
        // TODO Implement MediumInterface or check if it's even needed
        return nullptr;
    }

    bool isValid() const {
        // If this is a valid interaction inside a medium it is assumed that the
        // phase function has been initialized properly
        return phase != nullptr;
    }

   private:
    SurfacePoint surfacePoint;
    Vector3f wo;
    const Medium *medium;
    const PhaseFunction *phase;
};