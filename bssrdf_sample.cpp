#include "bssrdf_sample.h"
#include "scene.h"
#include "parallel.h"

struct bssrdf_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &isect = shading_isects[pixel_id];
        const auto &shape = scene.shapes[isect.shape_id];
        const auto &material = scene.materials[shape.material_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        const auto &shading_point = shading_points[pixel_id];

        next_rays[pixel_id] = Ray {
            shading_points[pixel_id].position,
            bssrdf_sample(
                  material,
                  shading_point,
                  -incoming_ray.dir,
                  bssrdf_samples[pixel_id],
                  min_roughness[pixel_id],
                  incoming_ray_differentials[pixel_id],
                  bssrdf_ray_differentials[pixel_id],
                  &next_min_roughness[pixel_id])};
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const Intersection *shading_isects;
    const SurfacePoint *shading_points;
    const BSSRDFSample *bssrdf_samples;
    const Real *min_roughness;
    Ray *next_rays;
    RayDifferential *bssrdf_ray_differentials;
    Real *next_min_roughness;
};

void bssrdf_sample(const Scene &scene,
                   const BufferView<int> &active_pixels,
                   const BufferView<Ray> &incoming_rays,
                   const BufferView<RayDifferential> &incoming_ray_differentials,
                   const BufferView<Intersection> &shading_isects,
                   const BufferView<SurfacePoint> &shading_points,
                   const BufferView<BSSRDFSample> &bssrdf_samples,
                   const BufferView<Real> &min_roughness,
                   BufferView<Ray> next_rays,
                   BufferView<RayDifferential> bssrdf_ray_differentials,
                   BufferView<Real> next_min_roughness) {
    parallel_for(
        bssrdf_sampler{get_flatten_scene(scene),
                       active_pixels.begin(),
                       incoming_rays.begin(),
                       incoming_ray_differentials.begin(),
                       shading_isects.begin(),
                       shading_points.begin(),
                       bssrdf_samples.begin(),
                       min_roughness.begin(),
                       next_rays.begin(),
                       bssrdf_ray_differentials.begin(),
                       next_min_roughness.begin()},
                       active_pixels.size(), scene.use_gpu);
}