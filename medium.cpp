#include "medium.h"
#include "parallel.h"
#include "scene.h"
#include "thrust_utils.h"

// !!!!TODO: This is incorrect when there is a medium inside a medium.
//           We need a stack to keep track of previous medium when we
//           exit a medium.
/**
 * Sample the given medium to see if the ray is affected by it. The
 * transmittance is encoded in the returned Vector3.
 */
DEVICE
inline
Vector3 sample(const Medium &medium,
               const Ray &ray,
               const Intersection &surface_isect,
               const MediumSample &sample,
               Intersection *medium_isect,
               Vector3 *medium_point) {
    if (medium.type == MediumType::homogeneous) {
        auto h = medium.homogeneous;
        // Sample a channel and distance along the ray
        auto channel = (int)min(int(sample.uv[0] * 3), 2);
        Real dist = -log(max(1 - sample.uv[1], Real(1e-20))) / h.sigma_t[channel];
        auto inside_medium = dist < ray.tmax;
        auto t = min(dist, ray.tmax);
        // Compute the transmittance and sampling density
        auto tr = exp(-h.sigma_t * min(t, MaxFloat));

        // Return the weighting factor for scattering inside of a homogeneous medium
        auto density = inside_medium ? (h.sigma_t * tr) : tr;
        auto pdf = sum(density) / 3;
        // Update intersection data
        *medium_point = ray.org + ray.dir * t;
        if (inside_medium) {
            medium_isect->shape_id = medium_isect->tri_id = -1;
        } else {
            // !!!!TODO: This is incorrect when there is a medium inside a medium.
            //           We need a stack to keep track of previous medium when we
            //           exit a medium.
            assert(surface_isect.valid());
            *medium_isect = surface_isect;
        }
        return inside_medium ? (tr * h.sigma_s / pdf) : (tr / pdf);
    } else {
        return Vector3{0, 0, 0};
    }
}

struct medium_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &medium_sample = medium_samples[pixel_id];
        const auto &surface_isect = surface_isects[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        auto &medium_isect = medium_isects[pixel_id];
        auto &medium_point = medium_points[pixel_id];
        if (medium_isect.medium_id >= 0) {
            // We are inside a medium. Sample a distance and compute transmittance.
            // Also update the intersection point.
            auto transmittance = sample(
                scene.mediums[medium_isect.medium_id],
                incoming_ray,
                surface_isect,
                medium_sample,
                &medium_isect,
                &medium_point);
            throughputs[pixel_id] *= transmittance;
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Intersection *surface_isects;
    const Ray *incoming_rays;
    const MediumSample *medium_samples;
    Intersection *medium_isects;
    Vector3 *medium_points;
    Vector3 *throughputs;
};

void sample_medium(const Scene &scene,
                   const BufferView<int> &active_pixels,
                   const BufferView<Intersection> &surface_isects,
                   const BufferView<Ray> &incoming_rays,
                   const BufferView<MediumSample> &medium_samples,
                   BufferView<Intersection> medium_isects,
                   BufferView<Vector3> medium_points,
                   BufferView<Vector3> throughputs) {
    parallel_for(medium_sampler{
        get_flatten_scene(scene),
        active_pixels.begin(),
        surface_isects.begin(),
        incoming_rays.begin(),
        medium_samples.begin(),
        medium_isects.begin(),
        medium_points.begin(),
        throughputs.begin()},
        active_pixels.size(), scene.use_gpu);
}

DEVICE
inline
Vector3 transmittance(const Medium &medium,
                      const Ray &ray,
                      const MediumSample &sample) {
    if (medium.type == MediumType::homogeneous) {
        auto h = medium.homogeneous;
        // Use Beer's Law to calculate transmittance
        return exp(-h.sigma_t * min(ray.tmax, MaxFloat));
    } else {
        return Vector3{0, 0, 0};
    }
}

struct transmittance_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        transmittances[pixel_id] = Vector3{1, 1, 1};
        if (medium_isects[pixel_id].medium_id >= 0) {
            const Ray &ray = rays[pixel_id];
            if (ray.tmax > 0) { // maxt <= 0 means the ray is occluded.
                const MediumSample &sample = samples[pixel_id];
                transmittances[pixel_id] = transmittance(
                    mediums[medium_isects[pixel_id].medium_id], ray, sample);
            }
        }
    }

    const Medium *mediums;
    const int *active_pixels;
    const Ray *rays;
    const Intersection *medium_isects;
    const MediumSample *samples;
    Vector3 *transmittances;
};

void evaluate_transmisttance(const Scene &scene,
                             const BufferView<int> &active_pixels,
                             const BufferView<Ray> &rays,
                             const BufferView<Intersection> &medium_isects,
                             const BufferView<MediumSample> &medium_samples,
                             BufferView<Vector3> transmittances) {
    // TODO: Currently we assume the first surface/volume we hit has a different
    // index of refraction or is fully opaque. This can be inefficient for some
    // cases, e.g. if we have an object with IOR=1 blocking the light.
    // To resolve this we want to repeatedly intersect with the scene until
    // we reach the end of the rays or we hit a surface with different IOR or is opaque.
    parallel_for(transmittance_sampler{
        scene.mediums.data,
        active_pixels.begin(),
        rays.begin(),
        medium_isects.begin(),
        medium_samples.begin(),
        transmittances.begin()},
        active_pixels.size(), scene.use_gpu);
}
