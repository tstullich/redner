#pragma once

#include "medium.h"

/**
 * An interface class to hold information about medium
 * boundaries. The medium could have different properties
 * on the inside and on the surface or it could even be
 * unbounded
 */
struct MediumInterface {
   public:
    // If only one type of medium is passed into the constructor we
    // assume that the medium is the same for the inside and outside
    MediumInterface(const Medium *medium) : inside(medium), outside(medium){};

    MediumInterface(const Medium *inside, const Medium *outside)
        : inside(inside), outside(outside){};

    // Determines if the inside and outside medium are the same
    bool isMediumTransition() const {
        return inside != outside;
    }

    const Medium *inside, *outside;
};