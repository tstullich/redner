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
                              // Shading point information
                              const BufferView<Ray> &incoming_rays,
                              const BufferView<Intersection> &surface_isects,
                              const BufferView<SurfacePoint> &surface_points,
                              const BufferView<int> &medium_ids,
                              const BufferView<Real> &medium_distances,
                              // Next event estimation information
                              const BufferView<Intersection> &light_isects,
                              const BufferView<SurfacePoint> &light_points,
                              const BufferView<Ray> &light_rays,
                              const BufferView<Intersection> &light_int_isects,
                              const BufferView<SurfacePoint> &light_int_points,
                              const BufferView<Ray> &light_int_rays,
                              const BufferView<int> &light_medium_ids,
                              // Scattering information
                              const BufferView<Intersection> &scatter_isects,
                              const BufferView<SurfacePoint> &scatter_points,
                              const BufferView<Ray> &scatter_rays,
                              const BufferView<Intersection> &next_isects,
                              const BufferView<SurfacePoint> &next_points,
                              const BufferView<Ray> &scatter_int_rays,
                              const BufferView<int> &scatter_medium_ids,
                              // Misc
                              const BufferView<Real> &min_roughness,
                              const Real weight,
                              const ChannelInfo &channel_info,
                              // Output
                              BufferView<Vector3> next_throughputs,
                              float *rendered_image,
                              BufferView<Real> edge_contribs);

/// The backward version of the function above.
void d_accumulate_path_contribs(const Scene &scene,
                                const BufferView<int> &active_pixels,
                                const BufferView<Vector3> &throughputs,
                                const BufferView<Vector3> &path_transmittances,
                                // Shading point information
                                const BufferView<Ray> &incoming_rays,
                                const BufferView<LightSample> &light_samples,
                                const BufferView<ScatterSample> &scatter_samples,
                                const BufferView<MediumSample> &medium_samples,
                                const BufferView<Intersection> &surface_isects,
                                const BufferView<SurfacePoint> &surface_points,
                                const BufferView<int> &medium_ids,
                                const BufferView<Real> &medium_distances,
                                // Next event estimation information
                                const BufferView<Intersection> &light_isects,
                                const BufferView<SurfacePoint> &light_points,
                                const BufferView<Ray> &light_rays,
                                const BufferView<Intersection> &light_int_isects,
                                const BufferView<SurfacePoint> &light_int_points,
                                const BufferView<Ray> &light_int_rays,
                                const BufferView<int> &light_medium_ids,
                                // Scattering information
                                const BufferView<Intersection> &scatter_isects,
                                const BufferView<SurfacePoint> &scatter_points,
                                const BufferView<Ray> &scatter_rays,
                                const BufferView<RayDifferential> &scatter_ray_differentials,
                                const BufferView<Intersection> &next_isects,
                                const BufferView<SurfacePoint> &next_points,
                                const BufferView<Ray> &scatter_int_rays,
                                const BufferView<int> &scatter_medium_ids,
                                // Misc
                                const BufferView<Real> &min_roughness,
                                const Real weight,
                                const ChannelInfo &channel_info,
                                // d_outputs
                                const float *d_rendered_image,
                                const BufferView<Vector3> &d_next_throughputs,
                                const BufferView<DRay> &d_next_rays,
                                const BufferView<RayDifferential> &d_next_ray_differentials,
                                const BufferView<SurfacePoint> &d_next_points,
                                DScene *d_scene,
                                BufferView<Vector3> d_throughputs,
                                BufferView<DRay> d_incoming_rays,
                                BufferView<SurfacePoint> d_surface_points);
