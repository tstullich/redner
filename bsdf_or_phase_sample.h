#pragma once

#include "redner.h"
#include "buffer.h"
#include "directional_sample.h"
#include "intersection.h"
#include "material.h"
#include "phase_function.h"
#include "ray.h"

struct Scene;

/**
 * Given incoming rays with intersected surfaces or participating media,
 * sample the next rays based on the material or phase function.
 * The backward pass of this function is computed at d_accumulate_path_contribs
 */
void bsdf_or_phase_sample(
        const Scene &scene,
        const BufferView<int> &active_pixels,
        const BufferView<Ray> &incoming_rays,
        const BufferView<RayDifferential> &incoming_ray_differentials,
        const BufferView<Intersection> &surface_isects,
        const BufferView<SurfacePoint> &surface_points,
        const BufferView<Intersection> &medium_isects,
        const BufferView<Vector3> &medium_points,
        const BufferView<DirectionalSample> &directional_samples,
        const BufferView<Real> &min_roughness,
        BufferView<Ray> next_rays,
        BufferView<RayDifferential> bsdf_ray_differentials,
        BufferView<Real> next_min_roughness);
