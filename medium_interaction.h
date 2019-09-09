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
    MediumInteraction(const SurfacePoint &p, const Vector3d &wo,
                      const Medium *medium, const PhaseFunction *phase)
        : surface_point(p), wo(wo), medium(medium), phase(phase){};

    bool valid() const {
        // If this is a valid interaction inside a medium it is assumed that the
        // phase function has been initialized properly
        return phase != nullptr;
    }

    SurfacePoint surface_point;
    Vector3f wo;
    const Medium *medium;
    const PhaseFunction *phase;
};