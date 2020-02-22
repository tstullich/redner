#include "primary_contribution.h"
#include "scene.h"
#include "channels.h"
#include "parallel.h"

struct primary_contribs_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &throughput = throughputs[pixel_id];
        const auto &surface_isect = surface_isects[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        Vector3 emission = Vector3{0, 0, 0};
        if (surface_isect.valid()) {
            const auto &surface_point = surface_points[pixel_id];
            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
            auto wi = -incoming_ray.dir;
            if (surface_shape.light_id >= 0) {
                const auto &light = scene.area_lights[surface_shape.light_id];
                if (light.directly_visible) {
                    if (light.two_sided || dot(wi, surface_point.shading_frame.n) > 0) {
                        emission += light.intensity;
                    }
                }
            }
        } else if (scene.envmap != nullptr) {
            if (scene.envmap->directly_visible) {
                auto dir = incoming_rays[pixel_id].dir;
                emission = envmap_eval(*(scene.envmap), dir, incoming_ray_differentials[pixel_id]);
            }
        }
        auto tr = Vector3{1, 1, 1};
        if (medium_ids != nullptr) {
            const auto &medium = scene.mediums[medium_ids[pixel_id]];
            tr = transmittance(medium, incoming_rays[pixel_id]);
        }
        auto contrib = weight * throughput * emission * tr;
        if (rendered_image != nullptr) {
            auto nc = channel_info.num_channels;
            auto nd = channel_info.num_total_dimensions;
            auto d = 0;
            for (int c = 0; c < nc; c++) {
                switch (channel_info.channels[c]) {
                    case Channels::radiance: {
                        rendered_image[nd * pixel_id + d] += contrib[0];
                        d++;
                        rendered_image[nd * pixel_id + d] += contrib[1];
                        d++;
                        rendered_image[nd * pixel_id + d] += contrib[2];
                        d++;
                    } break;
                    case Channels::alpha: {
                        if (surface_isect.valid()) {
                            auto alpha = weight;
                            if (channel_multipliers != nullptr) {
                                alpha *= channel_multipliers[nd * pixel_id + d];
                            }
                            rendered_image[nd * pixel_id + d] += alpha;
                        }
                        d++;
                    } break;
                    case Channels::depth: {
                        if (surface_isect.valid()) {
                            const auto &surface_point = surface_points[pixel_id];
                            auto depth = distance(incoming_ray.org,
                                                  surface_point.position) * weight;
                            if (channel_multipliers != nullptr) {
                                depth *= channel_multipliers[nd * pixel_id + d];
                            }
                            rendered_image[nd * pixel_id + d] += depth;
                        }
                        d++;
                    } break;
                    case Channels::position: {
                        if (surface_isect.valid()) {
                            const auto &surface_point = surface_points[pixel_id];
                            auto position = surface_point.position * weight;
                            if (channel_multipliers != nullptr) {
                                position[0] *= channel_multipliers[nd * pixel_id + d];
                                position[1] *= channel_multipliers[nd * pixel_id + d + 1];
                                position[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            }
                            rendered_image[nd * pixel_id + d    ] += position[0];
                            rendered_image[nd * pixel_id + d + 1] += position[1];
                            rendered_image[nd * pixel_id + d + 2] += position[2];
                        }
                        d += 3;
                    } break;
                    case Channels::geometry_normal: {
                        if (surface_isect.valid()) {
                            const auto &surface_point = surface_points[pixel_id];
                            auto geom_normal = surface_point.geom_normal * weight;
                            if (channel_multipliers != nullptr) {
                                geom_normal[0] *= channel_multipliers[nd * pixel_id + d];
                                geom_normal[1] *= channel_multipliers[nd * pixel_id + d + 1];
                                geom_normal[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            }
                            rendered_image[nd * pixel_id + d    ] += geom_normal[0];
                            rendered_image[nd * pixel_id + d + 1] += geom_normal[1];
                            rendered_image[nd * pixel_id + d + 2] += geom_normal[2];
                        }
                        d += 3;
                    } break;
                    case Channels::shading_normal: {
                        if (surface_isect.valid()) {
                            const auto &surface_point = surface_points[pixel_id];
                            auto shading_normal = surface_point.shading_frame[2] * weight;
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            const auto &material = scene.materials[surface_shape.material_id];
                            if (has_normal_map(material)) {
                                auto frame = perturb_shading_frame(material, surface_point);
                                shading_normal = frame.n * weight;
                            }
                            if (channel_multipliers != nullptr) {
                                shading_normal[0] *= channel_multipliers[nd * pixel_id + d];
                                shading_normal[1] *= channel_multipliers[nd * pixel_id + d + 1];
                                shading_normal[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            }
                            rendered_image[nd * pixel_id + d    ] += shading_normal[0];
                            rendered_image[nd * pixel_id + d + 1] += shading_normal[1];
                            rendered_image[nd * pixel_id + d + 2] += shading_normal[2];
                        }
                        d += 3;
                    } break;
                    case Channels::uv: {
                        if (surface_isect.valid()) {
                            const auto &surface_point = surface_points[pixel_id];
                            auto uv = surface_point.uv * weight;
                            if (channel_multipliers != nullptr) {
                                uv[0] *= channel_multipliers[nd * pixel_id + d];
                                uv[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            }
                            rendered_image[nd * pixel_id + d    ] += uv[0];
                            rendered_image[nd * pixel_id + d + 1] += uv[1];
                        }
                        d += 2;
                    } break;
                    case Channels::diffuse_reflectance: {
                        if (surface_isect.valid()) {
                            const auto &surface_point = surface_points[pixel_id];
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            const auto &material = scene.materials[surface_shape.material_id];
                            auto refl = material.use_vertex_color ?
                                surface_point.color : get_diffuse_reflectance(material, surface_point);
                            refl *= weight;
                            if (channel_multipliers != nullptr) {
                                refl[0] *= channel_multipliers[nd * pixel_id + d];
                                refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                                refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            }
                            rendered_image[nd * pixel_id + d    ] += refl[0];
                            rendered_image[nd * pixel_id + d + 1] += refl[1];
                            rendered_image[nd * pixel_id + d + 2] += refl[2];
                        }
                        d += 3;
                    } break;
                    case Channels::specular_reflectance: {
                        if (surface_isect.valid()) {
                            const auto &surface_point = surface_points[pixel_id];
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            const auto &material = scene.materials[surface_shape.material_id];
                            auto refl =
                                get_specular_reflectance(material, surface_point) * weight;
                            if (channel_multipliers != nullptr) {
                                refl[0] *= channel_multipliers[nd * pixel_id + d];
                                refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                                refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            }
                            rendered_image[nd * pixel_id + d    ] += refl[0];
                            rendered_image[nd * pixel_id + d + 1] += refl[1];
                            rendered_image[nd * pixel_id + d + 2] += refl[2];
                        }
                        d += 3;
                    } break;
                    case Channels::roughness: {
                        if (surface_isect.valid()) {
                            const auto &surface_point = surface_points[pixel_id];
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            const auto &material = scene.materials[surface_shape.material_id];
                            auto r = get_roughness(material, surface_point) * weight;
                            if (channel_multipliers != nullptr) {
                                r *= channel_multipliers[nd * pixel_id + d];
                            }
                            rendered_image[nd * pixel_id + d    ] += r;
                        }
                        d++;
                    } break;
                    case Channels::generic_texture: {
                        if (surface_isect.valid()) {
                            const auto &surface_point = surface_points[pixel_id];
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            const auto &material = scene.materials[surface_shape.material_id];
                            Real *buffer = &generic_texture_buffer[
                                scene.max_generic_texture_dimension * pixel_id];
                            get_generic_texture(material, surface_point, buffer);
                            for (int i = 0; i < material.generic_texture.channels; i++) {
                                auto gt = buffer[i];
                                gt *= weight;
                                if (channel_multipliers != nullptr) {
                                    gt *= channel_multipliers[nd * pixel_id + d + i];
                                }
                                rendered_image[nd * pixel_id + d + i] += gt;
                            }
                        }
                        d += scene.max_generic_texture_dimension;
                    } break;
                    case Channels::vertex_color: {
                        if (surface_isect.valid()) {
                            const auto &surface_point = surface_points[pixel_id];
                            auto refl = surface_point.color * weight;
                            if (channel_multipliers != nullptr) {
                                refl[0] *= channel_multipliers[nd * pixel_id + d];
                                refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                                refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            }
                            rendered_image[nd * pixel_id + d    ] += refl[0];
                            rendered_image[nd * pixel_id + d + 1] += refl[1];
                            rendered_image[nd * pixel_id + d + 2] += refl[2];
                        }
                        d += 3;
                    } break;
                    // when there are multiple samples per pixel,
                    // we use the last sample for determining the ids
                    case Channels::shape_id: {
                        if (surface_isect.valid()) {
                            rendered_image[nd * pixel_id + d    ] = Real(surface_isect.shape_id);
                        }
                        d++;
                    } break;
                    case Channels::triangle_id: {
                        if (surface_isect.valid()) {
                            rendered_image[nd * pixel_id + d    ] = Real(surface_isect.tri_id);
                        }
                        d++;
                    } break;
                    case Channels::material_id: {
                        if (surface_isect.valid()) {
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            rendered_image[nd * pixel_id + d    ] = Real(surface_shape.material_id);
                        }
                        d++;
                    } break;
                    default: {
                        assert(false);
                    }
                }
            }
        }
        if (edge_contribs != nullptr) {
            auto nc = channel_info.num_channels;
            auto nd = channel_info.num_total_dimensions;
            auto d = 0;
            for (int c = 0; c < nc; c++) {
                switch (channel_info.channels[c]) {
                    case Channels::radiance: {
                        edge_contribs[pixel_id] += sum(contrib);
                        d += 3;
                    } break;
                    case Channels::alpha: {
                        if (surface_isect.valid() && channel_multipliers != nullptr) {
                            auto alpha = weight;
                            alpha *= channel_multipliers[nd * pixel_id + d];
                            edge_contribs[pixel_id] += alpha;
                        }
                        d++;
                    } break;
                    case Channels::depth: {
                        if (surface_isect.valid() && channel_multipliers != nullptr) {
                            const auto &surface_point = surface_points[pixel_id];
                            auto depth = distance(incoming_ray.org,
                                                  surface_point.position) * weight;
                            depth *= channel_multipliers[nd * pixel_id + d];
                            edge_contribs[pixel_id] += depth;
                        }
                        d++;
                    } break;
                    case Channels::position: {
                        if (surface_isect.valid() && channel_multipliers != nullptr) {
                            const auto &surface_point = surface_points[pixel_id];
                            auto position = surface_point.position * weight;
                            position[0] *= channel_multipliers[nd * pixel_id + d];
                            position[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            position[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            edge_contribs[pixel_id] += sum(position);
                        }
                        d += 3;
                    } break;
                    case Channels::geometry_normal: {
                        if (surface_isect.valid() && channel_multipliers != nullptr) {
                            const auto &surface_point = surface_points[pixel_id];
                            auto geom_normal = surface_point.geom_normal * weight;
                            geom_normal[0] *= channel_multipliers[nd * pixel_id + d];
                            geom_normal[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            geom_normal[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            edge_contribs[pixel_id] += sum(geom_normal);
                        }
                        d += 3;
                    } break;
                    case Channels::shading_normal: {
                        if (surface_isect.valid() && channel_multipliers != nullptr) {
                            const auto &surface_point = surface_points[pixel_id];
                            auto shading_normal = surface_point.shading_frame[2] * weight;
                            shading_normal[0] *= channel_multipliers[nd * pixel_id + d];
                            shading_normal[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            shading_normal[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            edge_contribs[pixel_id] += sum(shading_normal);
                        }
                        d += 3;
                    } break;
                    case Channels::uv: {
                        if (surface_isect.valid() && channel_multipliers != nullptr) {
                            const auto &surface_point = surface_points[pixel_id];
                            auto uv = surface_point.uv * weight;
                            uv[0] *= channel_multipliers[nd * pixel_id + d];
                            uv[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            edge_contribs[pixel_id] += sum(uv);
                        }
                        d += 2;
                    } break;
                    case Channels::diffuse_reflectance: {
                        if (surface_isect.valid() && channel_multipliers != nullptr) {
                            const auto &surface_point = surface_points[pixel_id];
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            const auto &material = scene.materials[surface_shape.material_id];
                            auto refl = material.use_vertex_color ?
                                surface_point.color : get_diffuse_reflectance(material, surface_point);
                            refl *= weight;
                            refl[0] *= channel_multipliers[nd * pixel_id + d];
                            refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            edge_contribs[pixel_id] += sum(refl);
                        }
                        d += 3;
                    } break;
                    case Channels::specular_reflectance: {
                        if (surface_isect.valid() && channel_multipliers != nullptr) {
                            const auto &surface_point = surface_points[pixel_id];
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            const auto &material = scene.materials[surface_shape.material_id];
                            auto refl =
                                get_specular_reflectance(material, surface_point) * weight;
                            refl[0] *= channel_multipliers[nd * pixel_id + d];
                            refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            edge_contribs[pixel_id] += sum(refl);
                        }
                        d += 3;
                    } break;
                    case Channels::roughness: {
                        if (surface_isect.valid() && channel_multipliers != nullptr) {
                            const auto &surface_point = surface_points[pixel_id];
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            const auto &material = scene.materials[surface_shape.material_id];
                            auto r = get_roughness(material, surface_point) * weight;
                            r *= channel_multipliers[nd * pixel_id + d];
                            edge_contribs[pixel_id] += r;
                        }
                        d++;
                    } break;
                    case Channels::generic_texture: {
                        if (surface_isect.valid() && channel_multipliers != nullptr) {
                            const auto &surface_point = surface_points[pixel_id];
                            const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                            const auto &material = scene.materials[surface_shape.material_id];
                            Real *buffer = &generic_texture_buffer[
                                scene.max_generic_texture_dimension * pixel_id];
                            get_generic_texture(material, surface_point, buffer);
                            for (int i = 0; i < material.generic_texture.channels; i++) {
                                auto gt = buffer[i];
                                gt *= weight;
                                gt *= channel_multipliers[nd * pixel_id + d + i];
                                edge_contribs[pixel_id] += gt;
                            }
                        }
                        d += scene.max_generic_texture_dimension;
                    } break;
                    case Channels::vertex_color: {
                        if (surface_isect.valid() && channel_multipliers != nullptr) {
                            const auto &surface_point = surface_points[pixel_id];
                            auto refl = surface_point.color * weight;
                            refl[0] *= channel_multipliers[nd * pixel_id + d];
                            refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            edge_contribs[pixel_id] += sum(refl);
                        }
                        d += 3;
                    } break;
                    // ids are not differentiable
                    case Channels::shape_id: {
                        d++;
                    } break;
                    case Channels::triangle_id: {
                        d++;
                    } break;
                    case Channels::material_id: {
                        d++;
                    } break;
                    default: {
                        assert(false);
                    }
                }
            }
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Vector3 *throughputs;
    const Real *channel_multipliers;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const Intersection *surface_isects;
    const SurfacePoint *surface_points;
    const int *medium_ids;
    const Real weight;
    const ChannelInfo channel_info;
    float *rendered_image;
    Real *edge_contribs;
    Real *generic_texture_buffer;
};

struct d_primary_contribs_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &throughput = throughputs[pixel_id];
        const auto &surface_isect = surface_isects[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        auto nc = channel_info.num_channels;
        auto nd = channel_info.num_total_dimensions;
        auto d = 0;
        for (int c = 0; c < nc; c++) {
            switch (channel_info.channels[c]) {
                case Channels::radiance: {
                    auto tr = Vector3{1, 1, 1};
                    if (medium_ids != nullptr) {
                        const auto &medium = scene.mediums[medium_ids[pixel_id]];
                        tr = transmittance(medium, incoming_rays[pixel_id]);
                    }
                    // contrib = weight * throughput * emission * transmittance
                    auto d_emission = weight * throughput * tr *
                            Vector3{d_rendered_image[nd * pixel_id + d],
                                    d_rendered_image[nd * pixel_id + d + 1],
                                    d_rendered_image[nd * pixel_id + d + 2]};
                    auto emission = Vector3{0, 0, 0};
                    if (surface_isect.valid()) {
                        const auto &surface_point = surface_points[pixel_id];
                        const auto &surface_shape = scene.shapes[surface_isect.shape_id];
                        auto wi = -incoming_rays[pixel_id].dir;

                        if (surface_shape.light_id >= 0 &&
                                dot(wi, surface_point.shading_frame.n) > 0) {
                            const auto &light = scene.area_lights[surface_shape.light_id];
                            if (light.directly_visible) {
                                if (light.two_sided || dot(wi, surface_point.shading_frame.n) > 0) {
                                    emission = light.intensity;
                                    atomic_add(d_area_lights[surface_shape.light_id].intensity, d_emission);
                                }
                            }
                        }
                    } else if (scene.envmap != nullptr) {
                        if (scene.envmap->directly_visible) {
                            // Environment map
                            auto dir = incoming_rays[pixel_id].dir;
                            emission = envmap_eval(*(scene.envmap),
                                                   dir,
                                                   incoming_ray_differentials[pixel_id]);
                            d_envmap_eval(*(scene.envmap),
                                          dir,
                                          incoming_ray_differentials[pixel_id],
                                          d_emission,
                                          *d_envmap,
                                          d_incoming_rays[pixel_id].dir,
                                          d_incoming_ray_differentials[pixel_id]);
                        }
                    }
                    if (medium_ids != nullptr && sum(emission) > 0) {
                        // contrib = weight * throughput * emission * transmittance
                        auto d_tr = weight * throughput * emission *
                            Vector3{d_rendered_image[nd * pixel_id + d],
                                    d_rendered_image[nd * pixel_id + d + 1],
                                    d_rendered_image[nd * pixel_id + d + 2]};
                        // transmittance = transmittance(medium, incoming_rays[pixel_id]);
                        const auto &medium = scene.mediums[medium_ids[pixel_id]];
                        Ray d_r;
                        d_transmittance(medium, incoming_rays[pixel_id], d_tr,
                                        d_mediums[medium_ids[pixel_id]], d_r);
                        d_incoming_rays[pixel_id].org += d_r.org;
                        d_incoming_rays[pixel_id].dir += d_r.dir;
                        // tmax = distance(incoming_ray.org, surface_point.position)
                        d_distance(incoming_rays[pixel_id].org,
                                   surface_points[pixel_id].position,
                                   d_r.tmax,
                                   d_incoming_rays[pixel_id].org,
                                   d_surface_points[pixel_id].position);
                    }
                    d += 3;
                } break;
                case Channels::alpha: {
                    // Nothing to backprop to
                    d += 1;
                } break;
                case Channels::depth: {
                    if (surface_isect.valid()) {
                        const auto &surface_point = surface_points[pixel_id];
                        // auto depth = distance(incoming_ray.org,
                        //                       surface_point.position) * weight;
                        // depth *= channel_multipliers[nd * pixel_id + d];
                        auto d_depth = d_rendered_image[nd * pixel_id + d];
                        auto d_dist = d_depth * weight;
                        if (channel_multipliers != nullptr) {
                            d_dist *= channel_multipliers[nd * pixel_id + d];
                        }
                        auto d_org = Vector3{0, 0, 0};
                        auto d_position = Vector3{0, 0, 0};
                        d_distance(incoming_ray.org, surface_point.position, d_dist,
                                   d_org, d_position);
                        d_incoming_rays[pixel_id].org += d_org;
                        d_surface_points[pixel_id].position += d_position;
                    }
                    d += 1;
                } break;
                case Channels::position: {
                    if (surface_isect.valid()) {
                        auto d_position = weight *
                                Vector3{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1],
                                        d_rendered_image[nd * pixel_id + d + 2]};
                        if (channel_multipliers != nullptr) {
                            d_position[0] *= channel_multipliers[nd * pixel_id + d];
                            d_position[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            d_position[2] *= channel_multipliers[nd * pixel_id + d + 2];
                        }
                        d_surface_points[pixel_id].position += d_position;
                    }
                    d += 3;
                } break;
                case Channels::geometry_normal: {
                    if (surface_isect.valid()) {
                        auto d_geom_normal = weight *
                                Vector3{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1],
                                        d_rendered_image[nd * pixel_id + d + 2]};
                        if (channel_multipliers != nullptr) {
                            d_geom_normal[0] *= channel_multipliers[nd * pixel_id + d];
                            d_geom_normal[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            d_geom_normal[2] *= channel_multipliers[nd * pixel_id + d + 2];
                        }
                        d_surface_points[pixel_id].geom_normal += d_geom_normal;
                    }
                    d += 3;
                } break;
                case Channels::shading_normal: {
                    if (surface_isect.valid()) {
                        auto d_shading_normal = weight *
                                Vector3{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1],
                                        d_rendered_image[nd * pixel_id + d + 2]};
                        if (channel_multipliers != nullptr) {
                            d_shading_normal[0] *= channel_multipliers[nd * pixel_id + d];
                            d_shading_normal[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            d_shading_normal[2] *= channel_multipliers[nd * pixel_id + d + 2];
                        }
                        d_surface_points[pixel_id].shading_frame[2] += d_shading_normal;
                    }
                    d += 3;
                } break;
                case Channels::uv: {
                    if (surface_isect.valid()) {
                        auto d_uv = weight *
                                Vector2{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1]};
                        if (channel_multipliers != nullptr) {
                            d_uv[0] *= channel_multipliers[nd * pixel_id + d];
                            d_uv[1] *= channel_multipliers[nd * pixel_id + d + 1];
                        }
                        d_surface_points[pixel_id].uv += d_uv;
                    }
                    d += 2;
                } break;
                case Channels::diffuse_reflectance: {
                    if (surface_isect.valid()) {
                        const auto &surface_point = surface_points[pixel_id];
                        const auto &shape = scene.shapes[surface_isect.shape_id];
                        const auto &material = scene.materials[shape.material_id];
                        auto d_refl = weight *
                                Vector3{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1],
                                        d_rendered_image[nd * pixel_id + d + 2]};
                        if (channel_multipliers != nullptr) {
                            d_refl[0] *= channel_multipliers[nd * pixel_id + d];
                            d_refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            d_refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                        }
                        if (material.use_vertex_color) {
                            d_surface_points[pixel_id].color += d_refl;
                        } else {
                            d_get_diffuse_reflectance(material, surface_point, d_refl,
                                d_materials[shape.material_id].diffuse_reflectance,
                                d_surface_points[pixel_id]);
                        }
                    }
                    d += 3;
                } break;
                case Channels::specular_reflectance: {
                    if (surface_isect.valid()) {
                        const auto &surface_point = surface_points[pixel_id];
                        const auto &shape = scene.shapes[surface_isect.shape_id];
                        const auto &material = scene.materials[shape.material_id];
                        auto d_refl = weight *
                                Vector3{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1],
                                        d_rendered_image[nd * pixel_id + d + 2]};
                        if (channel_multipliers != nullptr) {
                            d_refl[0] *= channel_multipliers[nd * pixel_id + d];
                            d_refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            d_refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                        }
                        d_get_specular_reflectance(material, surface_point, d_refl,
                            d_materials[shape.material_id].specular_reflectance,
                            d_surface_points[pixel_id]);
                    }
                    d += 3;
                } break;
                case Channels::roughness: {
                    if (surface_isect.valid()) {
                        const auto &surface_point = surface_points[pixel_id];
                        const auto &shape = scene.shapes[surface_isect.shape_id];
                        const auto &material = scene.materials[shape.material_id];
                        auto d_roughness = weight *
                                d_rendered_image[nd * pixel_id + d];
                        if (channel_multipliers != nullptr) {
                            d_roughness *= channel_multipliers[nd * pixel_id + d];
                        }
                        d_get_roughness(material, surface_point, d_roughness,
                            d_materials[shape.material_id].roughness,
                            d_surface_points[pixel_id]);
                    }
                    d += 1;
                } break;
                case Channels::generic_texture: {
                    if (surface_isect.valid()) {
                        const auto &surface_point = surface_points[pixel_id];
                        const auto &shape = scene.shapes[surface_isect.shape_id];
                        const auto &material = scene.materials[shape.material_id];
                        Real *buffer = &generic_texture_buffer[
                            scene.max_generic_texture_dimension * pixel_id];
                        for (int i = 0; i < material.generic_texture.channels; i++) {
                            buffer[i] = weight * d_rendered_image[nd * pixel_id + d + i];
                            if (channel_multipliers != nullptr) {
                                buffer[i] *= channel_multipliers[nd * pixel_id + d + i];
                            }
                        }
                        d_get_generic_texture(material,
                                              surface_point,
                                              buffer,
                                              d_materials[shape.material_id].generic_texture,
                                              d_surface_points[pixel_id]);
                    }
                    d += scene.max_generic_texture_dimension;
                } break;
                case Channels::vertex_color: {
                    if (surface_isect.valid()) {
                        auto d_refl = weight *
                                Vector3{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1],
                                        d_rendered_image[nd * pixel_id + d + 2]};
                        if (channel_multipliers != nullptr) {
                            d_refl[0] *= channel_multipliers[nd * pixel_id + d];
                            d_refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            d_refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                        }
                        d_surface_points[pixel_id].color += d_refl;
                    }
                    d += 3;
                } break;
                // ids are not differentiable
                case Channels::shape_id: {
                    d++;
                } break;
                case Channels::triangle_id: {
                    d++;
                } break;
                case Channels::material_id: {
                    d++;
                } break;
                default: {
                    assert(false);
                }
            }
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Vector3 *throughputs;
    const Real *channel_multipliers;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const Intersection *surface_isects;
    const SurfacePoint *surface_points;
    const int *medium_ids;
    const Real weight;
    const ChannelInfo channel_info;
    const float *d_rendered_image;
    Real *generic_texture_buffer;
    DAreaLight *d_area_lights;
    DEnvironmentMap *d_envmap;
    DMaterial *d_materials;
    DMedium *d_mediums;
    DRay *d_incoming_rays;
    RayDifferential *d_incoming_ray_differentials;
    SurfacePoint *d_surface_points;
};

void accumulate_primary_contribs(
        const Scene &scene,
        const BufferView<int> &active_pixels,
        const BufferView<Vector3> &throughputs,
        const BufferView<Real> &channel_multipliers,
        const BufferView<Ray> &incoming_rays,
        const BufferView<RayDifferential> &incoming_ray_differentials,
        const BufferView<Intersection> &surface_isects,
        const BufferView<SurfacePoint> &surface_points,
        const BufferView<int> &medium_ids,
        const Real weight,
        const ChannelInfo &channel_info,
        float *rendered_image,
        BufferView<Real> edge_contribs,
        BufferView<Real> generic_texture_buffer) {
    parallel_for(primary_contribs_accumulator{
        get_flatten_scene(scene),
        active_pixels.begin(),
        throughputs.begin(),
        channel_multipliers.begin(),
        incoming_rays.begin(),
        incoming_ray_differentials.begin(),
        surface_isects.begin(),
        surface_points.begin(),
        medium_ids.begin(),
        weight,
        channel_info,
        rendered_image,
        edge_contribs.begin(),
        generic_texture_buffer.begin()
    }, active_pixels.size(), scene.use_gpu);
}

void d_accumulate_primary_contribs(
        const Scene &scene,
        const BufferView<int> &active_pixels,
        const BufferView<Vector3> &throughputs,
        const BufferView<Real> &channel_multipliers,
        const BufferView<Ray> &incoming_rays,
        const BufferView<RayDifferential> &incoming_ray_differentials,
        const BufferView<Intersection> &surface_isects,
        const BufferView<SurfacePoint> &surface_points,
        const BufferView<int> &medium_ids,
        const Real weight,
        const ChannelInfo &channel_info,
        const float *d_rendered_image,
        BufferView<Real> generic_texture_buffer,
        DScene *d_scene,
        BufferView<DRay> d_incoming_rays,
        BufferView<RayDifferential> d_incoming_ray_differentials,
        BufferView<SurfacePoint> d_surface_points) {
    parallel_for(d_primary_contribs_accumulator{
        get_flatten_scene(scene),
        active_pixels.begin(),
        throughputs.begin(),
        channel_multipliers.begin(),
        incoming_rays.begin(),
        incoming_ray_differentials.begin(),
        surface_isects.begin(),
        surface_points.begin(),
        medium_ids.begin(),
        weight,
        channel_info,
        d_rendered_image,
        generic_texture_buffer.begin(),
        d_scene->area_lights.data,
        d_scene->envmap,
        d_scene->materials.data,
        d_scene->mediums.data,
        d_incoming_rays.begin(),
        d_incoming_ray_differentials.begin(),
        d_surface_points.begin()
    }, active_pixels.size(), scene.use_gpu);
}
