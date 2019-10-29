import pyredner
import torch

pyredner.set_use_gpu(False)
#pyredner.set_use_gpu(torch.cuda.is_available())

cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, -5.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

mat_grey = pyredner.Material(\
    diffuse_reflectance = \
        torch.tensor([0.5, 0.5, 0.5], device = pyredner.get_device()))

# The material list of the scene
materials = [mat_grey]

shape_triangle = pyredner.Shape(\
    vertices = torch.tensor([[-1.7, 1.0, 0.0], [1.0, 1.0, 0.0], [-0.5, -1.0, 0.0]],
        device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2]], dtype = torch.int32,
        device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0,
    medium = pyredner.HomogeneousMedium(torch.tensor([2.19, 2.62, 3.00]),
        torch.tensor([0.0021, 0.0041, 0.0071]), torch.tensor([0.7])))

shape_light = pyredner.Shape(\
    vertices = torch.tensor([[-1.0, -1.0, -7.0],
                             [ 1.0, -1.0, -7.0],
                             [-1.0,  1.0, -7.0],
                             [ 1.0,  1.0, -7.0]], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)

# The shape list of our scene contains two shapes:
shapes = [shape_triangle, shape_light]

light = pyredner.AreaLight(shape_id = 1,
                           intensity = torch.tensor([20.0,20.0,20.0]))
area_lights = [light]
# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam, shapes, materials, area_lights)
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

render = pyredner.RenderFunction.apply
img = render(0, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_medium/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_medium/target.png')