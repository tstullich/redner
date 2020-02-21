#pragma once

#include "redner.h"
#include "vector.h"

template <typename T>
struct TScatterSample {
    TVector2<T> uv;
    T w;
};

using ScatterSample = TScatterSample<Real>;
