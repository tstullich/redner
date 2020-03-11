import pyredner
import torch

pyredner.set_use_gpu(torch.cuda.is_available())

# Intitialize information about the medium. We can set the
# absorption as well as scattering factors for the medium.
# g is a parameter that pertains to the phase function that is going
# to be used. Redner currently only supports the Henyey-Greenstein
# phase function, but it should be possible to add others in the future
mediums = [pyredner.HomogeneousMedium( \
    sigma_a = torch.tensor([0.005, 0.005, 0.005]),
    sigma_s = torch.tensor([0.00001, 0.00001, 0.00001]),
    g = torch.tensor([0.0]))]

# Attach medium information to the camera to get a fog effect
# throughout the whole scene
#cam = pyredner.Camera(position = torch.tensor([313, 221.7, 97.034]),
#                      look_at = torch.tensor([0.0, 221.7, 0.0]),
cam = pyredner.Camera(position = torch.tensor([22.84, 2.37, -1.8]),
                      look_at = torch.tensor([7.0, 1.082, 7.66]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([70.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (512, 512),
                      medium_id = 0)

mat_light = pyredner.Material( \
    diffuse_reflectance = \
        torch.tensor([1.0, 1.0, 1.0], device = pyredner.get_device()))

materials = [mat_light]

print('Loading San Miguel model')
material_map, mesh_list, light_map = pyredner.load_obj('models/san-miguel/san-miguel.obj')

# Setup material map
material_id_map = {}
count = len(materials)
for key, value in material_map.items():
    material_id_map[key] = count
    count += 1
    materials.append(value)

shapes = []
for mtl_name, mesh in mesh_list:
    shapes.append(pyredner.Shape(
        vertices = mesh.vertices,
        indices = mesh.indices,
        material_id = material_id_map[mtl_name],
        normals = mesh.normals,
        exterior_medium_id = 0))

print('Generating light source')

light = pyredner.generate_quad_light(torch.tensor([15.291, 12.0, -4.47]),
    torch.tensor([16.09, 3.878, -4.89]),
    torch.tensor([30.0, 30.0]),
    torch.tensor([10.0, 10.0, 10.0]))

# Shape describing the light. In this case we use an area light source
# facing downward onto the scene
shape_light = pyredner.Shape(
    vertices = light.vertices,
    indices = light.indices,
    material_id = 0,
    exterior_medium_id = 0)
shapes.insert(0, shape_light)

light = pyredner.AreaLight(shape_id = 0,
                           intensity = light.light_intensity,
                           two_sided=True)
area_lights = [light]

# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam,
                       shapes,
                       materials,
                       area_lights,
                       mediums = mediums)
scene_args = pyredner.RenderFunction.serialize_scene( \
    scene = scene,
    num_samples = 512,
    max_bounces = 2)

render = pyredner.RenderFunction.apply
target = render(0, *scene_args)
pyredner.imwrite(target.cpu(), 'results/test_medium_miguel/target.exr')
pyredner.imwrite(target.cpu(), 'results/test_medium_miguel/target.png')
target = pyredner.imread('results/test_medium_miguel/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Before perturbing save values
sigma_a_val = mediums[0].sigma_a.clone().detach()
sigma_s_val = mediums[0].sigma_s.clone().detach()

# Perturb the medium for the initial guess.
# Here we set the absorption factor to be optimized.
# A higher medium absorption factor corresponds to less light being
# transmitted so the goal is to move from a darkened image
# to a lighter one.
mediums[0].sigma_a = torch.tensor( \
    [0.2, 0.2, 0.2],
    device = pyredner.get_device(),
    requires_grad = True)

mediums[0].sigma_s = torch.tensor( \
    [0.2, 0.2, 0.2],
    device = pyredner.get_device(),
    requires_grad = True)

## Serialize scene arguments
scene_args = pyredner.RenderFunction.serialize_scene( \
    scene = scene,
    num_samples = 512,
    max_bounces = 2)

## Render initial guess
img = render(1, *scene_args)
## Save image
pyredner.imwrite(img.cpu(), 'results/test_medium_miguel/init.png')
## Compute the difference between the target and initial guess
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_medium_miguel/init_diff.png')

# Setup loss csv file
with open('results/test_medium_miguel/miguel-loss.csv', 'w') as file:
    file.write('a b')
    file.write('\n')

# Setup param diff file
with open('results/test_medium_miguel/miguel-param.csv', 'w') as file:
    file.write('a b c')
    file.write('\n')

# Optimize absorption factor of medium inside the sphere
optimizer = torch.optim.Adam([mediums[0].sigma_a, mediums[0].sigma_s], lr=5e-3)
# Run Adam for 100 iterations
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass to render the image
    scene_args = pyredner.RenderFunction.serialize_scene( \
        scene = scene,
        num_samples = 512,
        max_bounces = 2)

    # Use a different seed per iteration
    img = render(t + 1, *scene_args)
    pyredner.imwrite(img.cpu(), 'results/test_medium_miguel/iter_{}.png'.format(t))

    # Compute the loss function
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    with open('results/test_medium_miguel/miguel-loss.csv', 'a') as file:
        file.write(str(t) + ' ' + str(loss.item()))
        file.write('\n')

    with open('results/test_medium_miguel/miguel-param.csv', 'a') as file:
        file.write(str(t) + ' ')
        file.write(str(torch.abs(sigma_a_val - mediums[0].sigma_a).sum().item()) + ' ')
        file.write(str(torch.abs(sigma_s_val - mediums[0].sigma_s).sum().item()))
        file.write('\n')

    # Backpropagate the gradients
    loss.backward()
    # Print the gradients of the absorption factor
    print('grad0:', mediums[0].sigma_a.grad)
    print('grad1:', mediums[0].sigma_s.grad)

    # Take a gradient descent step
    optimizer.step()
    # Clamp sigma_a to a valid value
    mediums[0].sigma_a.data.clamp_(0.00001)
    mediums[0].sigma_s.data.clamp_(0.00001)
    # Print the current absorption factor values
    print('sigma_a:', mediums[0].sigma_a)
    print('sigma_s:', mediums[0].sigma_s)

with open('results/test_medium_miguel/final_val.txt', 'w') as file:
    file.write(str(mediums[0].sigma_a))
    file.write('\n')
    file.write(str(mediums[0].sigma_s))
    file.write('\n')

# Render final result
scene_args = pyredner.RenderFunction.serialize_scene( \
    scene = scene,
    num_samples = 512,
    max_bounces = 2)
img = render(102, *scene_args)

# Save the images and diffs
pyredner.imwrite(img.cpu(), 'results/test_medium_miguel/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_medium_miguel/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_medium_miguel/final_diff.png')