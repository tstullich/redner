#pragma once

#include "redner.h"
#include "vector.h"

template <typename T>
struct TDirectionalSample {
    TVector2<T> uv;
    T w;
};

using DirectionalSample = TDirectionalSample<Real>;
