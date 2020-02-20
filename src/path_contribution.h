#pragma once

#include "redner.h"
#include "buffer.h"
#include "ray.h"
#include "intersection.h"
#include "shape.h"
#include "texture.h"
#include "area_light.h"
#include "material.h"
#include "medium.h"

struct Scene;
struct ChannelInfo;
struct DScene;

/// Compute the contribution at a path vertex, by combining next event estimation & BSDF sampling.
void accumulate_path_contribs(const Scene &scene,
                              const BufferView<int> &active_pixels,
                              const BufferView<Vector3> &throughputs,
                              const BufferView<Vector3> &path_transmittances,
                              const BufferView<Vector3> &nee_transmittences,
                              const BufferView<Vector3> &directional_transmittences,
                              const BufferView<Ray> &incoming_rays,
                              const BufferView<Intersection> &surface_isects,
                              const BufferView<SurfacePoint> &surface_points,
                              const BufferView<int> &medium_ids,
                              const BufferView<Real> &medium_distances,
                              const BufferView<Intersection> &light_isects,
                              const BufferView<SurfacePoint> &light_points,
                              const BufferView<Ray> &light_rays,
                              const BufferView<Intersection> &directional_isects,
                              const BufferView<SurfacePoint> &directional_points,
                              const BufferView<Ray> &directional_rays,
                              const BufferView<Real> &min_roughness,
                              const Real weight,
                              const ChannelInfo &channel_info,
                              BufferView<Vector3> next_throughputs,
                              float *rendered_image,
                              BufferView<Real> edge_contribs);

/// The backward version of the function above.
void d_accumulate_path_contribs(const Scene &scene,
                                const BufferView<int> &active_pixels,
                                const BufferView<Vector3> &throughputs,
                                const BufferView<Vector3> &path_transmittances,
                                const BufferView<Vector3> &nee_transmittances,
                                const BufferView<Vector3> &directional_transmittances,
                                const BufferView<Ray> &incoming_rays,
                                const BufferView<RayDifferential> &ray_differentials,
                                const BufferView<LightSample> &light_samples,
                                const BufferView<DirectionalSample> &directional_samples,
                                const BufferView<Intersection> &surface_isects,
                                const BufferView<SurfacePoint> &surface_points,
                                const BufferView<int> &medium_ids,
                                const BufferView<Real> &medium_distances,
                                const BufferView<Intersection> &light_isects,
                                const BufferView<SurfacePoint> &light_points,
                                const BufferView<Ray> &light_rays,
                                const BufferView<Intersection> &directional_isects,
                                const BufferView<SurfacePoint> &directional_points,
                                const BufferView<Ray> &directional_rays,
                                const BufferView<Real> &min_roughness,
                                const Real weight,
                                const ChannelInfo &channel_info,
                                const float *d_rendered_image,
                                const BufferView<Vector3> &d_next_throughputs,
                                BufferView<DRay> d_next_rays,
                                const BufferView<SurfacePoint> &d_next_points,
                                DScene *d_scene,
                                BufferView<Vector3> d_throughputs,
                                BufferView<Vector3> d_path_transmittances,
                                BufferView<Vector3> d_nee_transmittances,
                                BufferView<Vector3> d_directional_transmittances,
                                BufferView<DRay> d_incoming_rays,
                                BufferView<RayDifferential> d_incoming_ray_differentials,
                                BufferView<SurfacePoint> d_shading_points);
