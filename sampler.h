#pragma once

#include "buffer.h"
#include "area_light.h"
#include "camera.h"
#include "directional_sample.h"
#include "edge.h"
#include "material.h"
#include "medium.h"
#include "vector.h"

struct Sampler {
    virtual ~Sampler() {}
    virtual void begin_sample(int sample_id) {};

    virtual void next_camera_samples(BufferView<TCameraSample<float>> samples) = 0;
    virtual void next_camera_samples(BufferView<TCameraSample<double>> samples) = 0;
    virtual void next_light_samples(BufferView<TLightSample<float>> samples) = 0;
    virtual void next_light_samples(BufferView<TLightSample<double>> samples) = 0;
    virtual void next_directional_samples(BufferView<TDirectionalSample<float>> samples) = 0;
    virtual void next_directional_samples(BufferView<TDirectionalSample<double>> samples) = 0;
    virtual void next_primary_edge_samples(BufferView<TPrimaryEdgeSample<float>> samples) = 0;
    virtual void next_primary_edge_samples(BufferView<TPrimaryEdgeSample<double>> samples) = 0;
    virtual void next_secondary_edge_samples(BufferView<TSecondaryEdgeSample<float>> samples) = 0;
    virtual void next_secondary_edge_samples(BufferView<TSecondaryEdgeSample<double>> samples) = 0;
    virtual void next_medium_samples(BufferView<TMediumSample<float>> samples) = 0;
    virtual void next_medium_samples(BufferView<TMediumSample<double>> samples) = 0;
};
