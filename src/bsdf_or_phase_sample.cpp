#include "bsdf_or_phase_sample.h"
#include "medium.h"
#include "parallel.h"
#include "phase_function.h"
#include "scene.h"

struct bsdf_or_phase_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &incoming_ray = incoming_rays[pixel_id];
        const auto &medium_isect = medium_isects[pixel_id];

        if (medium_isect.medium_id >= 0) {
            // Sample phase function for new direction
            next_rays[pixel_id] = Ray{
                medium_points[pixel_id],
                sample_phase_function(
                    get_phase_function(scene.mediums[medium_isect.medium_id]),
                    -incoming_ray.dir,
                    directional_samples[pixel_id])};
            // Update ray differentials: currently we treat it like diffuse surface
            // TODO: figure out better way to propagate ray differentials,
            // ideally covariance tracing.
            bsdf_ray_differentials[pixel_id].org_dx = incoming_ray_differentials[pixel_id].org_dx;
            bsdf_ray_differentials[pixel_id].org_dy = incoming_ray_differentials[pixel_id].org_dy;
            bsdf_ray_differentials[pixel_id].dir_dx = Vector3{0.03f, 0.03f, 0.03f};
            bsdf_ray_differentials[pixel_id].dir_dy = Vector3{0.03f, 0.03f, 0.03f};
        } else {
            // Sample BSDF
            const auto &surface_isect = surface_isects[pixel_id];
            assert(surface_isect.valid());
            const auto &shape = scene.shapes[surface_isect.shape_id];
            const auto &material = scene.materials[shape.material_id];
            const auto &surface_point = surface_points[pixel_id];
            next_rays[pixel_id] = Ray{
                surface_point.position,
                bsdf_sample(
                    material,
                    surface_point,
                    -incoming_ray.dir,
                    directional_samples[pixel_id],
                    min_roughness[pixel_id],
                    incoming_ray_differentials[pixel_id],
                    bsdf_ray_differentials[pixel_id],
                    &next_min_roughness[pixel_id])};
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const Intersection *surface_isects;
    const SurfacePoint *surface_points;
    const Intersection *medium_isects;
    const Vector3 *medium_points;
    const DirectionalSample *directional_samples;
    const Real *min_roughness;
    Ray *next_rays;
    RayDifferential *bsdf_ray_differentials;
    Real *next_min_roughness;
};

void bsdf_or_phase_sample(const Scene &scene,
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
                          BufferView<Real> next_min_roughness) {
    parallel_for(
        bsdf_or_phase_sampler{get_flatten_scene(scene),
                              active_pixels.begin(),
                              incoming_rays.begin(),
                              incoming_ray_differentials.begin(),
                              surface_isects.begin(),
                              surface_points.begin(),
                              medium_isects.begin(),
                              medium_points.begin(),
                              directional_samples.begin(),
                              min_roughness.begin(),
                              next_rays.begin(),
                              bsdf_ray_differentials.begin(),
                              next_min_roughness.begin()},
                              active_pixels.size(), scene.use_gpu);
}
