import pyredner
import torch

pyredner.set_use_gpu(torch.cuda.is_available())

# Intitialize information about the medium. We can set the
# absorption as well as scattering factors for the medium.
# g is a parameter that pertains to the phase function that is going
# to be used. Redner currently only supports the Henyey-Greenstein
# phase function, but it should be possible to add others in the future
mediums = [pyredner.HomogeneousMedium( \
    sigma_a = torch.tensor([0.05, 0.05, 0.05]),
    sigma_s = torch.tensor([0.00001, 0.5, 0.5]),
    g = torch.tensor([0.0])),

    pyredner.HomogeneousMedium(\
        sigma_a = torch.tensor([0.0001, 0.0001, 0.0001]),
        sigma_s = torch.tensor([0.5, 0.00001, 0.00001]),
        g = torch.tensor([0.9]))]

# Attach medium information to the camera to get a fog effect
# throughout the whole scene
cam = pyredner.Camera(position = torch.tensor([-1.0, 0.5, 0.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([70.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (512, 512),
                      medium_id = 1)

# The materials for the scene - one for the sphere and one for the
# surrounding planes. The light source emits white light
mat_sphere = pyredner.Material( \
    diffuse_reflectance = \
        torch.tensor([0.89, 0.15, 0.21], device = pyredner.get_device()))

mat_light = pyredner.Material( \
    diffuse_reflectance = \
        torch.tensor([1.0, 1.0, 1.0], device = pyredner.get_device()))

mat_planes = pyredner.Material( \
    diffuse_reflectance = \
        torch.tensor([0.0, 0.19, 0.56], device = pyredner.get_device()))

materials = [mat_sphere, mat_light, mat_planes]

print('Loading dragon model')
# Setup for various objects in the scene
material_map, mesh_list, light_map = pyredner.load_obj('models/dragon.obj')

light = pyredner.generate_quad_light(torch.tensor([-2.0, 1.0, 0.0]),
    torch.tensor([0.0, 0.0, 0.0]),
    torch.tensor([5.0, 5.0]),
    torch.tensor([1.0, 1.0, 1.0]))

shapes = []
# Shape describing the light. In this case we use an area light source
# facing downward onto the scene
shape_light = pyredner.Shape( \
    vertices = light.vertices,
    indices = light.indices,
    uvs = None,
    normals = None,
    material_id = 0,
    interior_medium_id = -1,
    exterior_medium_id = 1)
shapes.append(shape_light)

for mtl_name, mesh in mesh_list:
    shapes.append(pyredner.Shape( \
        vertices = mesh.vertices,
        indices = mesh.indices,
        material_id = -1,
        normals = mesh.normals,
        interior_medium_id = 0,
        exterior_medium_id = 1))

# Shape describing the floor
shape_floor = pyredner.Shape( \
    vertices = torch.tensor([[7.0,  -1.5,  5.0],
                             [7.0,  -1.5, -5.0],
                             [-7.0, -1.5, -5.0],
                             [-7.0, -1.5,  5.0]],
                            device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[0, 2, 3]],
                           dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 2,
    interior_medium_id = -1,
    exterior_medium_id = 1)
shapes.append(shape_floor)

# Shape describing the backplane
shape_back = pyredner.Shape( \
    vertices = torch.tensor([[8.0,  -9.0, -5.0],
                             [8.0,   7.0, -5.0],
                             [-8.0,  7.0, -5.0],
                             [-8.0, -9.0, -5.0]],
                            device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[0, 2, 3]],
                           dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 2,
    interior_medium_id = -1,
    exterior_medium_id = 1)
shapes.append(shape_back)

# Shape describing the left side of the box
shape_left = pyredner.Shape( \
    vertices = torch.tensor([[-5.0,  -1.5,  5.0],
                             [-5.0,   6.0,  5.0],
                             [-5.0,   6.0, -5.0],
                             [-5.0,  -1.5, -5.0]],
                            device = pyredner.get_device()),
    indices = torch.tensor([[2, 1, 0],[3, 2, 0]],
                           dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 2,
    interior_medium_id = -1,
    exterior_medium_id = 1)
shapes.append(shape_left)

# Shape describing the right side of the box
shape_right = pyredner.Shape( \
    vertices = torch.tensor([[5.0,  -1.5,  5.0],
                             [5.0,   6.0,  5.0],
                             [5.0,   6.0, -5.0],
                             [5.0,  -1.5, -5.0]],
                            device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[0, 2, 3]],
                           dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 2,
    interior_medium_id = -1,
    exterior_medium_id = 1)
shapes.append(shape_right)

# Config 1 - A complete box + a sphere
#shapes = [shape_light, shape_sphere, shape_floor, shape_back, shape_left, shape_right]

light = pyredner.AreaLight(shape_id = 0,
                           intensity = light.light_intensity)
area_lights = [light]

# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam,
                       shapes,
                       materials,
                       area_lights,
                       mediums = mediums)
scene_args = pyredner.RenderFunction.serialize_scene( \
    scene = scene,
    num_samples = 256,
    max_bounces = 1)

render = pyredner.RenderFunction.apply
target = render(0, *scene_args)
pyredner.imwrite(target.cpu(), 'results/test_nested/target.exr')
pyredner.imwrite(target.cpu(), 'results/test_nested/target.png')
target = pyredner.imread('results/test_nested/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Before perturbing save old values
sigma_a_val = torch.tensor(mediums[0].sigma_a, device=pyredner.get_device())
hg_g_val = torch.tensor(mediums[1].g, device=pyredner.get_device())

# Perturb the medium for the initial guess.
# Here we set the absorption factor to be optimized.
# A higher medium absorption factor corresponds to less light being
# transmitted so the goal is to move from a darkened image
# to a lighter one.
mediums[0].sigma_a = torch.tensor( \
    [0.3, 0.3, 0.3],
    device = pyredner.get_device(),
    requires_grad = True)

mediums[1].g = torch.tensor( \
    [-0.9],
    device = pyredner.get_device(),
    requires_grad = True)

## Serialize scene arguments
scene_args = pyredner.RenderFunction.serialize_scene( \
    scene = scene,
    num_samples = 256,
    max_bounces = 1)

## Render initial guess
img = render(1, *scene_args)
## Save image
pyredner.imwrite(img.cpu(), 'results/test_nested/init.png')
## Compute the difference between the target and initial guess
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_nested/init_diff.png')

# Setup loss csv file
with open('results/test_nested/nested-loss.csv', 'w') as file:
    file.write('a b')
    file.write('\n')

# Setup diff file
with open('results/test_nested/nested-param.csv', 'w') as file:
    file.write('a b c')
    file.write('\n')

# Optimize absorption factor of medium inside the sphere
optimizer = torch.optim.Adam([mediums[0].sigma_a, mediums[1].g], lr=5e-3)
# Run Adam for 100 iterations
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass to render the image
    scene_args = pyredner.RenderFunction.serialize_scene( \
        scene = scene,
        num_samples = 256,
        max_bounces = 1)

    # Use a different seed per iteration
    img = render(t + 1, *scene_args)
    pyredner.imwrite(img.cpu(), 'results/test_nested/iter_{}.png'.format(t))

    # Compute the loss function
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    with open('results/test_nested/nested-loss.csv', 'a') as file:
        file.write(str(t) + ' ' + str(loss.item()))
        file.write('\n')

    with open('results/test_nested/nested-param.csv', 'a') as file:
        file.write(str(t) + ' ')
        file.write(str(torch.abs(sigma_a_val - mediums[0].sigma_a).sum().item()) + ' ')
        file.write(str(torch.abs(hg_g_val - mediums[1].g).sum().item()) + ' ')
        file.write(str(mediums[0].sigma_a) + ' ')
        file.write(str(mediums[1].g))
        file.write('\n')

    # Backpropagate the gradients
    loss.backward()
    # Print the gradients of the absorption factor
    print('grad 0:', mediums[0].sigma_a.grad)
    print('grad 1:', mediums[1].g.grad)

    # Take a gradient descent step
    optimizer.step()
    # Clamp sigma_a to a valid value
    mediums[0].sigma_a.data.clamp_(0.00001)
    mediums[1].g.data.clamp_(0.00001)
    # Print the current absorption factor values
    print('sigma_a:', mediums[0].sigma_a)
    print('g:', mediums[1].g)

with open('results/test_nested/final_val.txt', 'w') as file:
    file.write(str(mediums[0].sigma_a))
    file.write('\n')
    file.write(str(mediums[1].g))
    file.write('\n')

# Render final result
scene_args = pyredner.RenderFunction.serialize_scene( \
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
img = render(102, *scene_args)

# Save the images and diffs
pyredner.imwrite(img.cpu(), 'results/test_nested/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_nested/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_nested/final_diff.png')