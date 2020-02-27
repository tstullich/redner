
import pyredner
import torch
import redner

pyredner.set_use_gpu(torch.cuda.is_available())

# Intitialize information about the medium. We can set the
# absorption as well as scattering factors for the medium.
# g is a parameter that pertains to the phase function that is going
# to be used. Redner currently only supports the Henyey-Greenstein
# phase function, but it should be possible to add others in the future
mediums = [pyredner.HomogeneousMedium(\
    sigma_a = torch.tensor([0.05, 0.05, 0.05]),
    sigma_s = torch.tensor([0.00001, 0.00001, 0.00001]),
    g = torch.tensor([0.0]))]

# The materials for the scene - one for the sphere and one for the
# surrounding planes. The light source emits white light
mat_buddha = pyredner.Material(\
    diffuse_reflectance = \
        torch.tensor([0.89, 0.15, 0.21], device = pyredner.get_device()))

materials = [mat_buddha]

print('Loading buddha model')

# Setup for various objects in the scene
material_map, mesh_list, light_map = pyredner.load_obj('scenes/buddha/buddha.obj')
for _, mesh in mesh_list:
    print('Computing normals')
    mesh.normals = pyredner.compute_vertex_normal(mesh.vertices, mesh.indices)

shapes = []
for mtl_name, mesh in mesh_list:
    shapes.append(pyredner.Shape(\
        vertices = mesh.vertices,
        indices = mesh.indices,
        material_id = 0,
        normals = mesh.normals,
        interior_medium_id = -1,
        exterior_medium_id = -1))

envmap_img = pyredner.imread('venice_sunset.exr')
if pyredner.get_use_gpu():
    envmap = envmap_img.cuda(device = pyredner.get_device())
envmap = pyredner.EnvironmentMap(envmap_img)

cam = pyredner.automatic_camera_placement(shapes, (256, 256))

# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam,
                       shapes,
                       materials,
                       envmap = envmap,
                       mediums = mediums)
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 32,
    max_bounces = 2)

render = pyredner.RenderFunction.apply
target = render(0, *scene_args)
pyredner.imwrite(target.cpu(), 'results/test_medium_buddha/target.exr')
pyredner.imwrite(target.cpu(), 'results/test_medium_buddha/target.png')
target = pyredner.imread('results/test_medium_buddha/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

exit()

# Perturb the medium for the initial guess.
# Here we set the absorption factor to be optimized.
# A higher medium absorption factor corresponds to less light being
# transmitted so the goal is to move from a darkened image
# to a lighter one.
mediums[0].sigma_a = torch.tensor(\
    [0.3, 0.3, 0.3],
    device = pyredner.get_device(),
    requires_grad = True)

## Serialize scene arguments
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 2,
    # Disable edge sampling for now
    use_primary_edge_sampling = False,
    use_secondary_edge_sampling = False)

## Render initial guess
img = render(1, *scene_args)
## Save image
pyredner.imwrite(img.cpu(), 'results/test_medium_buddha/init.png')
## Compute the difference between the target and initial guess
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_medium_buddha/init_diff.png')

# Optimize absorption factor of medium inside the sphere
optimizer = torch.optim.Adam([mediums[0].sigma_a], lr=5e-2)
# Run Adam for 200 iterations
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass to render the image
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 256,
        max_bounces = 2,
        use_primary_edge_sampling = False,
        use_secondary_edge_sampling = False)

    # Use a different seed per iteration
    img = render(t + 1, *scene_args)
    pyredner.imwrite(img.cpu(), 'results/test_medium_buddha/iter_{}.png'.format(t))

    # Compute the loss function
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients
    loss.backward()
    # Print the gradients of the absorption factor
    print('grad:', mediums[0].sigma_a.grad)

    # Take a gradient descent step
    optimizer.step()
    # Clamp sigma_a to a valid value
    mediums[0].sigma_a.data.clamp_(0.00001)
    # Print the current absorption factor values
    print('sigma_a:', mediums[0].sigma_a)

# Render final result
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 2,
    use_primary_edge_sampling = False,
    use_secondary_edge_sampling = False)
img = render(202, *scene_args)

# Save the images and diffs
pyredner.imwrite(img.cpu(), 'results/test_medium_buddha/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_medium_buddha/final.png')
pyredner.imwrite(torch.abs(target - img), 'results/test_medium_buddha/final_diff.png')