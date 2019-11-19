import pyredner
import torch
import redner

pyredner.set_use_gpu(False)
#pyredner.set_use_gpu(torch.cuda.is_available())

mediums = [pyredner.HomogeneousMedium(\
    sigma_a = torch.tensor([0.085867, 0.18314, 0.25262]),
    #sigma_a = torch.tensor([0.061, 0.97, 1.45]),
    sigma_s = torch.tensor([0.011002, 0.010927, 0.011036]),
    #sigma_s = torch.tensor([0.18, 0.07, 0.03]),
    g = torch.tensor([0.5])),
    # Second medium
    pyredner.HomogeneousMedium(\
    sigma_a = torch.tensor([0.001, 0.001, 0.001]),
    sigma_s = torch.tensor([0.002, 0.002, 0.002]),
    g = torch.tensor([-0.7]))]

cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, 5.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([60.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      medium_id = 1)

mat_red = pyredner.Material(\
    diffuse_reflectance = \
        torch.tensor([0.5, 0.5, 0.5], device = pyredner.get_device()))

# The material list of the scene
materials = [mat_red]

sphere = pyredner.generate_sphere(128, 64)
shape_sphere = pyredner.Shape(\
    vertices = sphere[0],
    indices = sphere[1],
    uvs = sphere[2],
    normals = sphere[3],
    material_id = 0,
    medium_id = -1)

shape_light = pyredner.Shape(\
    vertices = torch.tensor([[-3.0,  1.0,  1.0],
                             [-3.0,  1.0, -1.0],
                             [-2.0,  3.0,  1.0],
                             [-2.0,  3.0, -1.0]],
                             device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)

# The shape list of our scene contains two shapes:
shapes = [shape_sphere, shape_light]

light = pyredner.AreaLight(shape_id = 1,
                           intensity = torch.tensor([20.0, 20.0, 20.0]))
area_lights = [light]

envmap = pyredner.imread('/home/tim/projects/master/redner/tests/sunsky.exr')
if pyredner.get_use_gpu():
    envmap = envmap.cuda(device = pyredner.get_device())
envmap = pyredner.EnvironmentMap(envmap)

# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam,
                       shapes,
                       materials,
                       area_lights,
                       envmap = envmap,
                       mediums = mediums)
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 2,
    sampler_type = redner.SamplerType.sobol)

render = pyredner.RenderFunction.apply
img = render(0, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_medium/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_medium/target.png')