#include "medium.h"
#include "parallel.h"
#include "scene.h"
#include "thrust_utils.h"

DEVICE
inline
Real sample_distance(const Medium &medium, const MediumSample &sample) {
    if (medium.type == MediumType::homogeneous) {
        // Sample a channel and distance along the ray
        auto h = medium.homogeneous;
        auto channel = min(int(sample.uv[0] * 3), 2);
        return Real(-log(max(1 - sample.uv[1], Real(1e-20))) / h.sigma_t[channel]);
    } else {
        return Real(0);
    }
}

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
        Real dist = sample_distance(medium, sample);
        auto inside_medium = dist < ray.tmax;
        auto t = min(dist, ray.tmax);
        // Compute the transmittance and sampling density
        auto tr = exp(-h.sigma_t * min(t, MaxFloat) * length(ray.dir));

        // Return the weighting factor for scattering inside of a homogeneous medium
        auto density = inside_medium ? (h.sigma_t * tr) : tr;
        auto pdf = sum(density) / 3;
        // Update intersection data
        *medium_point = ray.org + ray.dir * t;
        if (inside_medium) {
            // Set the triangle ID to -1 if we are inside a medium
            medium_isect->tri_id = -1;
        } else {
            // Left our current medium. Update intersection data accordingly
            if (medium_isect->prev_medium_id >= 0) {
                // We previously encountered a medium
                medium_isect->medium_id = medium_isect->prev_medium_id;
                medium_isect->shape_id = medium_isect->prev_shape_id;
                medium_isect->prev_medium_id = -1;
                medium_isect->prev_shape_id = -1;
                medium_isect->tri_id = -1;
            } else {
                // No prior intersection was made. Take the surface intersection
                assert(surface_isect.valid());
                *medium_isect = surface_isect;
            }
        }
        return inside_medium ? (tr * h.sigma_s / pdf) : (tr / pdf);
    } else if (medium.type == MediumType::heterogeneous) {
        return Vector3{0, 0, 0};
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
                      const Ray &ray) {
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
        const Ray &ray = rays[pixel_id];
        transmittances[pixel_id] = Vector3{1, 1, 1};
        // tmax <= 0 means the ray is occluded
        if (medium_isects[pixel_id].medium_id >= 0 && ray.tmax > 0) {
            transmittances[pixel_id] = transmittance(
                mediums[medium_isects[pixel_id].medium_id], ray);
        }
    }

    const Medium *mediums;
    const int *active_pixels;
    const Ray *rays;
    const Intersection *medium_isects;
    const MediumSample *samples;
    Vector3 *transmittances;
};

void evaluate_transmittance(const Scene &scene,
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

/**
 * Find the derivative of the transmittance. This requires
 * us to have the previously computed transmittances
 */
DEVICE
inline
Vector3 d_transmittance(const Medium &medium,
                        const Ray &ray,
                        const Vector3 &medium_isect,
                        const Real &t) {
    if (medium.type == MediumType::homogeneous) {
        auto h = medium.homogeneous;
        auto tr = transmittance(medium, ray);

        // Calculate change rate of sigma_t using the material derivative for sigma_t
        auto d_sig_t = (h.sigma_t * medium_isect) + (h.sigma_t * t * -ray.dir);
        auto d_x = ray.org - (t * ray.dir);
        auto grad_sig_t = Vector3{1, 1, 1}; // TODO For testing only. Need to add more logic here
        d_sig_t += dot(d_x, grad_sig_t);

        // Calculate the attenuation along the ray accounting for the
        // rate of change with regards to t
        auto d_t = Real(0); // TODO Need to find change rate for t
        auto attenutation = d_t * (h.sigma_t * (medium_isect - t * ray.org));

        // Calculate the transmittance again taking into account the partial derivative
        // of sigma_t
        auto tr_sig_t = exp(-d_sig_t * min(ray.tmax, MaxFloat));

        // Put together all of the components for the derivative of the transmittance
        // TODO Check if we need the minus sign for the first transmittance term
        auto d_tr = -tr * (tr_sig_t + attenutation);
        return d_tr;
    } else {
        return Vector3{0, 0, 0};
    }
}

struct d_medium_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &incoming_ray = incoming_rays[pixel_id];
        const auto &medium_isect = medium_isects[pixel_id];
        const auto &medium = scene.mediums[medium_isect.medium_id];
        const auto &medium_sample = medium_samples[pixel_id];
        const auto &medium_point = medium_points[pixel_id];

        auto dist = sample_distance(medium, medium_sample);
        if (dist < incoming_ray.tmax) {
            // Inside medium. Calculate volumetric derivatives
            // !!!! This introduces a discontinuity when finding the gradient !!!!
            auto t = min(dist, incoming_ray.tmax);

            // Compute the derivative of the transmittance term
            auto d_tr = d_transmittance(medium, incoming_ray, medium_point, t);

        } else {
            // TODO Implement calculating the "interfacial" term
            auto tr = transmittance(medium, incoming_ray);

            // Find the partial derivative of the transmittance
            // !!!! This introduces a discontinuity when finding the gradient !!!!
            auto t = min(dist, incoming_ray.tmax);
            auto d_tr = d_transmittance(medium, incoming_ray, medium_point, t);

            // Find the partial derivative of the surface reflectance
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Ray *incoming_rays;
    const MediumSample *medium_samples;
    const Intersection *medium_isects;
    const DirectionalSample *directional_samples;
    Vector3 *medium_points;
    Vector3 *throughputs;
};

void d_sample_medium(const Scene &scene,
                     const BufferView<int> &active_pixels,
                     const BufferView<Ray> &incoming_rays,
                     const BufferView<MediumSample> &medium_samples,
                     const BufferView<Intersection> medium_isects,
                     const BufferView<DirectionalSample> &directional_samples,
                     BufferView<Vector3> medium_points,
                     BufferView<Vector3> throughputs) {
    parallel_for(d_medium_sampler {
        get_flatten_scene(scene),
        active_pixels.begin(),
        incoming_rays.begin(),
        medium_samples.begin(),
        medium_isects.begin(),
        directional_samples.begin(),
        medium_points.begin(),
        throughputs.begin()},
        active_pixels.size(), scene.use_gpu);
}