import pyredner
import torch
import redner

pyredner.set_use_gpu(False)
#pyredner.set_use_gpu(torch.cuda.is_available())

mediums = [pyredner.HomogeneousMedium(\
    sigma_a = torch.tensor([0.085867, 0.18314, 0.25262]),
    sigma_s = torch.tensor([0.011002, 0.010927, 0.011036]),
    g = torch.tensor([0.9])),
    # Second medium
    pyredner.HomogeneousMedium(\
    sigma_a = torch.tensor([0.05, 0.05, 0.05]),
    sigma_s = torch.tensor([0.1, 0.1, 0.1]),
    g = torch.tensor([0.0]))]

cam = pyredner.Camera(position = torch.tensor([0.0, 0.5, 5.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([70.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      medium_id = 1)

# The materials for the scene - one for the sphere and one for the
# surrounding planes
mat_sphere = pyredner.Material(\
    diffuse_reflectance = \
        torch.tensor([0.07, 0.07, 0.07], device = pyredner.get_device()))

mat_planes = pyredner.Material(\
    diffuse_reflectance = \
        torch.tensor([0.6, 0.6, 0.9], device = pyredner.get_device()))

materials = [mat_sphere, mat_planes]

sphere = pyredner.generate_sphere(128, 64)
shape_sphere = pyredner.Shape(\
    vertices = sphere[0],
    indices = sphere[1],
    uvs = sphere[2],
    normals = sphere[3],
    material_id = 0,
    medium_id = -1)

# Manually translating sphere since redner does not seem to support
# geometric transformations
shape_sphere.vertices = shape_sphere.vertices + torch.tensor([-1.0, 1.0, 0.0])

shape_light = pyredner.Shape(\
    vertices = torch.tensor([[-5.0,  3.0,  1.0],
                             [-5.0,  3.0, -1.0],
                             [-4.0,  5.0,  1.0],
                             [-4.0,  5.0, -1.0]],
                             device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)

# Shape describing the floor
shape_floor = pyredner.Shape(\
    vertices = torch.tensor([[7.0,  -1.5,  5.0],
                             [7.0,  -1.5, -5.0],
                             [-7.0, -1.5, -5.0],
                             [-7.0, -1.5,  5.0]],
                             device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[0, 2, 3]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 1)

# Shape describing the backplane
shape_back = pyredner.Shape(\
    vertices = torch.tensor([[5.0,  -1.5, -5.0],
                             [5.0,   6.0, -5.0],
                             [-6.0,  6.0, -5.0],
                             [-6.0, -1.5, -5.0]],
                             device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[0, 2, 3]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 1)

# Shape describing the right side of the box
shape_right = pyredner.Shape(\
    vertices = torch.tensor([[5.0,  -1.5,  5.0],
                             [5.0,   6.0,  5.0],
                             [5.0,   6.0, -7.0],
                             [5.0,  -1.5, -7.0]],
                             device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[0, 2, 3]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 1)

# The shape list of our scene containing multiple shapes:
shapes = [shape_sphere, shape_light, shape_floor, shape_back, shape_right]

light = pyredner.AreaLight(shape_id = 1,
                           intensity = torch.tensor([10.0, 10.0, 10.0]))
area_lights = [light]

# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam,
                       shapes,
                       materials,
                       area_lights,
                       mediums = mediums)
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 5,
    sampler_type = redner.SamplerType.sobol)

render = pyredner.RenderFunction.apply
img = render(0, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_medium/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_medium/target.png')
target = pyredner.imread('results/test_medium/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Perturb the medium for the initial guess
# Here we set the absorption factor to be optimized
mediums[0].sigma_a = torch.tensor(\
    [1.0, 1.0, 1.0],
    device = pyredner.get_device(),
    requires_grad = True)

## Serialize scene arguments
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 1,
    # Disable edge sampling for now
    use_primary_edge_sampling = False,
    use_secondary_edge_sampling = False)

## Render initial guess
img = render(1, *scene_args)
## Save image
pyredner.imwrite(img.cpu(), 'results/test_medium/init.png')
## Compute the difference between the target and initial guess
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_medium/init_diff.png')