#include "medium.h"
#include "active_pixels.h"
#include "parallel.h"
#include "scene.h"
#include "thrust_utils.h"

struct medium_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &medium_sample = medium_samples[pixel_id];
        const auto &surface_isect = surface_isects[pixel_id];
        const auto &surface_point = surface_points[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        const auto &medium_id = medium_ids[pixel_id];
        auto &next_medium_id = next_medium_ids[pixel_id];
        auto &medium_distance = medium_distances[pixel_id];
        if (medium_id >= 0) {
            // We are inside a medium. Sample a distance and compute transmittance.
            // Also update the intersection point.
            transmittances[pixel_id] = sample_medium(
                scene.mediums,
                scene.shapes,
                medium_id,
                incoming_ray,
                surface_isect,
                surface_point,
                medium_sample,
                &next_medium_id,
                &medium_distance);
        } else {
            transmittances[pixel_id] = Vector3{1, 1, 1};
            next_medium_id = -1;
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Intersection *surface_isects;
    const SurfacePoint *surface_points;
    const Ray *incoming_rays;
    const MediumSample *medium_samples;
    const int *medium_ids;
    int *next_medium_ids;
    Real *medium_distances;
    Vector3 *transmittances;
};

void sample_medium(const Scene &scene,
                   const BufferView<int> &active_pixels,
                   const BufferView<Intersection> &surface_isects,
                   const BufferView<SurfacePoint> &surface_points,
                   const BufferView<Ray> &incoming_rays,
                   const BufferView<MediumSample> &medium_samples,
                   const BufferView<int> &medium_ids,
                   BufferView<int> next_medium_ids,
                   BufferView<Real> medium_distances,
                   BufferView<Vector3> transmittances) {
    parallel_for(medium_sampler{
        get_flatten_scene(scene),
        active_pixels.begin(),
        surface_isects.begin(),
        surface_points.begin(),
        incoming_rays.begin(),
        medium_samples.begin(),
        medium_ids.begin(),
        next_medium_ids.begin(),
        medium_distances.begin(),
        transmittances.begin()},
        active_pixels.size(), scene.use_gpu);
}

struct transmittance_shadow_ray_spawner {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &int_isect = int_isects[pixel_id];
        if (int_isect.valid()) {
            const auto &shape = shapes[int_isect.shape_id];
            if (shape.material_id != -1) {
                // Hit an opaque object
                // Spawn an invalid ray
                shadow_rays[pixel_id] = Ray{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                };
                int_medium_ids[pixel_id] = -1;
            } else {
                if (light_isects[pixel_id].valid()) {
                    // Hit a transparent object, advance the transmittance ray
                    auto p0 = int_points[pixel_id].position;
                    auto p1 = light_points[pixel_id].position;
                    shadow_rays[pixel_id] = Ray{
                        p0, // org
                        normalize(p1 - p0), // dir
                        Real(1e-3), // tmin
                        (1 - 1e-3f) * distance(p0, p1), // tmax
                    };
                } else {
                    // Environment map
                    auto p0 = int_points[pixel_id].position;
                    shadow_rays[pixel_id] = Ray{
                        p0, // org
                        prev_rays[pixel_id].dir, // dir
                        Real(1e-3), // tmin
                    };
                }
                // Update medium id
                const auto &p = int_points[pixel_id];
                const auto &shape = shapes[int_isect.shape_id];
                if (dot(shadow_rays[pixel_id].dir, p.geom_normal) < 0) {
                    int_medium_ids[pixel_id] = shape.exterior_medium_id;
                } else {
                    int_medium_ids[pixel_id] = shape.interior_medium_id;
                }
            }
        } else {
            // Hit nothing -- don't need to trace shadow ray
            shadow_rays[pixel_id] = Ray{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
            };
            int_medium_ids[pixel_id] = -1;
        }
    }

    const Shape *shapes;
    const int *active_pixels;
    const Intersection *int_isects;
    const SurfacePoint *int_points;
    const Intersection *light_isects;
    const SurfacePoint *light_points;
    const Ray *prev_rays;
    int *int_medium_ids;
    Ray *shadow_rays;
};

struct transmittance_ray_spawner {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &int_isect = int_isects[pixel_id];
        if (int_isect.valid()) {
            const auto &shape = shapes[int_isect.shape_id];
            if (shape.material_id != -1) {
                // Hit an opaque object
                // Spawn an invalid ray
                next_rays[pixel_id] = Ray{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                };
                int_medium_ids[pixel_id] = -1;
            } else {
                // Hit a transparent object, advance the transmittance ray
                next_rays[pixel_id] = Ray{
                    int_points[pixel_id].position, // org
                    prev_rays[pixel_id].dir,
                    Real(1e-3) // tmin
                };
                // Update medium id
                const auto &p = int_points[pixel_id];
                const auto &shape = shapes[int_isect.shape_id];
                if (dot(next_rays[pixel_id].dir, p.geom_normal) < 0) {
                    int_medium_ids[pixel_id] = shape.exterior_medium_id;
                } else {
                    int_medium_ids[pixel_id] = shape.interior_medium_id;
                }
            }
        } else {
            // Hit nothing -- don't need to trace new ray
            next_rays[pixel_id] = Ray{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
            };
            int_medium_ids[pixel_id] = -1;
        }
    }

    const Shape *shapes;
    const int *active_pixels;
    const Intersection *int_isects;
    const SurfacePoint *int_points;
    const Ray *prev_rays;
    int *int_medium_ids;
    Ray *next_rays;
};

struct transmittance_intersection_update {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &int_isect = int_isects[pixel_id];
        if (int_isect.valid()) {
            const auto &shape = shapes[int_isect.shape_id];
            if (shape.material_id != -1) {
                // Hit an opaque object
                // set light_isects to the same intersection
                light_isects[pixel_id] = int_isect;
                light_points[pixel_id] = int_points[pixel_id];
            } else {
                // Hit a transparent object
                // leave light_isects as is
            }
        } else {
            // Hit nothing -- set light_isects to invalid
            light_isects[pixel_id] = Intersection{-1, -1};
        }
    }

    const Shape *shapes;
    const int *active_pixels;
    const Intersection *int_isects;
    const SurfacePoint *int_points;
    Intersection *light_isects;
    SurfacePoint *light_points;
};

void trace_nee_transmittance_rays(const Scene &scene,
                                  const BufferView<int> &active_pixels,
                                  const BufferView<Ray> &outgoing_rays,
                                  const BufferView<int> &medium_ids,
                                  const BufferView<Intersection> &light_isects,
                                  const BufferView<SurfacePoint> &light_points,
                                  BufferView<Ray> int_rays,
                                  BufferView<Intersection> int_isects,
                                  BufferView<SurfacePoint> int_points,
                                  BufferView<int> int_medium_ids,
                                  ThrustCachedAllocator &thrust_alloc,
                                  BufferView<OptiXRay> optix_rays,
                                  BufferView<OptiXHit> optix_hits) {
    // Copy outgoing rays to intermediate rays
    DISPATCH(scene.use_gpu, thrust::copy,
        outgoing_rays.begin(), outgoing_rays.end(),
        int_rays.begin());
    // Intersect with the scene
    intersect(scene,
              active_pixels,
              int_rays,
              BufferView<RayDifferential>(),
              int_isects,
              int_points,
              BufferView<RayDifferential>(),
              optix_rays,
              optix_hits);
    // Update medium ids and spawn transmittance shadow ray
    parallel_for(transmittance_shadow_ray_spawner{
            scene.shapes.data,
            active_pixels.begin(),
            int_isects.begin(),
            int_points.begin(),
            light_isects.begin(),
            light_points.begin(),
            outgoing_rays.begin(),
            int_medium_ids.begin(),
            int_rays.begin()
        }, active_pixels.size(), scene.use_gpu);
    // Shadow ray
    occluded(scene, active_pixels, int_rays, optix_rays, optix_hits);
}

void trace_scatter_transmittance_rays(const Scene &scene,
                                      const BufferView<int> &active_pixels,
                                      BufferView<Ray> outgoing_rays,
                                      const BufferView<RayDifferential> &ray_differentials,
                                      const BufferView<int> &medium_ids,
                                      BufferView<Intersection> light_isects,
                                      BufferView<SurfacePoint> light_points,
                                      BufferView<Ray> int_rays,
                                      BufferView<RayDifferential> new_ray_differentials,
                                      BufferView<Intersection> int_isects,
                                      BufferView<SurfacePoint> int_points,
                                      BufferView<int> int_medium_ids,
                                      ThrustCachedAllocator &thrust_alloc,
                                      BufferView<OptiXRay> optix_rays,
                                      BufferView<OptiXHit> optix_hits) {
    // Intersect with the scene
    intersect(scene,
              active_pixels,
              outgoing_rays,
              ray_differentials,
              int_isects,
              int_points,
              new_ray_differentials,
              optix_rays,
              optix_hits);
    // Update medium ids and spawn transmittance shadow ray
    parallel_for(transmittance_ray_spawner{
            scene.shapes.data,
            active_pixels.begin(),
            int_isects.begin(),
            int_points.begin(),
            outgoing_rays.begin(),
            int_medium_ids.begin(),
            int_rays.begin()
        }, active_pixels.size(), scene.use_gpu);
    // Intersect with the scene again
    intersect(scene,
              active_pixels,
              int_rays,
              BufferView<RayDifferential>(),
              light_isects,
              light_points,
              BufferView<RayDifferential>(),
              optix_rays,
              optix_hits);
    // Update intersection
    parallel_for(transmittance_intersection_update{
            scene.shapes.data,
            active_pixels.begin(),
            int_isects.begin(),
            int_points.begin(),
            light_isects.begin(),
            light_points.begin(),
        }, active_pixels.size(), scene.use_gpu);
}
