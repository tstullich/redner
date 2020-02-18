#include "bsdf_or_phase_sample.h"
#include "medium.h"
#include "parallel.h"
#include "phase_function.h"
#include "scene.h"

struct bsdf_or_phase_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &incoming_ray = incoming_rays[pixel_id];
        // Determine whether we should sample bsdf or phase function
        // If we are on a surface, and the surface has material assigned,
        // we should sample bsdf
        // If we are not on a surface, we should sample phase function
        // If we are on a surface, and the surface has no material assigned,
        // we keep the ray direction, that is, we ignore the surface.
        auto sample_phase = false;
        if (medium_ids != nullptr && medium_distances != nullptr &&
              medium_ids[pixel_id] >= 0 &&
              medium_distances[pixel_id] < incoming_ray.tmax) {
            sample_phase = true;
        }
        if (sample_phase) {
            // We're in a medium, sample phase function for new direction
            next_rays[pixel_id] = Ray{
                incoming_ray.org + incoming_ray.dir * medium_distances[pixel_id],
                sample_phase_function(
                    get_phase_function(scene.mediums[medium_ids[pixel_id]]),
                    -incoming_ray.dir,
                    directional_samples[pixel_id])};
            // Update ray differentials: currently we treat it like diffuse surface
            // TODO: figure out better way to propagate ray differentials,
            // ideally covariance tracing.
            next_ray_differentials[pixel_id].org_dx = incoming_ray_differentials[pixel_id].org_dx;
            next_ray_differentials[pixel_id].org_dy = incoming_ray_differentials[pixel_id].org_dy;
            next_ray_differentials[pixel_id].dir_dx = Vector3{0.03f, 0.03f, 0.03f};
            next_ray_differentials[pixel_id].dir_dy = Vector3{0.03f, 0.03f, 0.03f};
        } else {
            // Sample BSDF
            const auto &surface_isect = surface_isects[pixel_id];
            if (!surface_isect.valid()) {
                std::cerr << "pixel_id:" << pixel_id << std::endl;
                std::cerr << "medium_id:" << medium_ids[pixel_id] << std::endl;
                std::cerr << "medium_distance:" << medium_distances[pixel_id] << std::endl;
                std::cerr << "incoming_ray.tmax:" << incoming_ray.tmax << std::endl;
            }
            assert(surface_isect.valid());
            const auto &shape = scene.shapes[surface_isect.shape_id];
            const auto &surface_point = surface_points[pixel_id];
            if (shape.material_id >= 0) {
                const auto &material = scene.materials[shape.material_id];
                next_rays[pixel_id] = Ray{
                    surface_point.position,
                    bsdf_sample(
                        material,
                        surface_point,
                        -incoming_ray.dir,
                        directional_samples[pixel_id],
                        min_roughness[pixel_id],
                        incoming_ray_differentials[pixel_id],
                        next_ray_differentials[pixel_id],
                        &next_min_roughness[pixel_id])};
            } else {
                // The surface is transparent with IOR=1, ignore it 
                // and keep going
                next_rays[pixel_id] = Ray{
                    surface_point.position, incoming_ray.dir};
                next_ray_differentials[pixel_id] =
                    incoming_ray_differentials[pixel_id];
                next_min_roughness[pixel_id] = min_roughness[pixel_id];
            }
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const Intersection *surface_isects;
    const SurfacePoint *surface_points;
    const int *medium_ids;
    const Real *medium_distances;
    const DirectionalSample *directional_samples;
    const Real *min_roughness;
    Ray *next_rays;
    RayDifferential *next_ray_differentials;
    Real *next_min_roughness;
};

void bsdf_or_phase_sample(const Scene &scene,
                          const BufferView<int> &active_pixels,
                          const BufferView<Ray> &incoming_rays,
                          const BufferView<RayDifferential> &incoming_ray_differentials,
                          const BufferView<Intersection> &surface_isects,
                          const BufferView<SurfacePoint> &surface_points,
                          const BufferView<int> &medium_ids,
                          const BufferView<Real> &medium_distances,
                          const BufferView<DirectionalSample> &directional_samples,
                          const BufferView<Real> &min_roughness,
                          BufferView<Ray> next_rays,
                          BufferView<RayDifferential> next_ray_differentials,
                          BufferView<Real> next_min_roughness) {
    parallel_for(
        bsdf_or_phase_sampler{get_flatten_scene(scene),
                              active_pixels.begin(),
                              incoming_rays.begin(),
                              incoming_ray_differentials.begin(),
                              surface_isects.begin(),
                              surface_points.begin(),
                              medium_ids.begin(),
                              medium_distances.begin(),
                              directional_samples.begin(),
                              min_roughness.begin(),
                              next_rays.begin(),
                              next_ray_differentials.begin(),
                              next_min_roughness.begin()},
                              active_pixels.size(), scene.use_gpu);
}
