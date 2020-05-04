import os
import torch

from utils.io import load_mesh

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturedSoftPhongShader,
    HardPhongShader,
    HardFlatShader
)


import os

# from pytorch3d.utils import image_grid

def render(mesh, device, renderer='flat'):
    if isinstance(mesh, str):
        # verts, faces, _ = load_obj(obj_filename, load_textures=False, device=device)
        verts, faces, = load_mesh(mesh)
        # faces = faces.verts_idx
    elif isinstance(mesh, list) or isinstance(mesh, tuple):
        verts = mesh[0]
        faces = mesh[1]
    else:
        raise ValueError("Unexpected mesh input of type '%s'. Pass in either a path to a mesh or its vertices "
                         "and faces in a list or tuple" % str(type(mesh)))

    # Load obj file

    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)

    verts_rgb[:,:,0] = 135/255
    verts_rgb[:,:,1] = 206/255
    verts_rgb[:,:,2] = 250/255
    #
    # verts_rgb[:,:,0] = 30/255
    # verts_rgb[:,:,1] = 206/255
    # verts_rgb[:,:,2] = 250/255

    # verts_rgb[:,:,0] = 0/255
    # verts_rgb[:,:,1] = 191/255
    # verts_rgb[:,:,2] = 255/255

    textures = Textures(verts_rgb=verts_rgb.to(device))
    mesh = Meshes([verts,], [faces,], textures)
    mesh = mesh.to(device)

    # Initialize an OpenGL perspective camera.
    batch_size = 5
    # elev = torch.linspace(0, 180, batch_size)
    azim = torch.linspace(-90, 90, batch_size)

    R, T = look_at_view_transform(0.35, elev=0, azim=azim,
                                  at=((0, -0.025, 0),),)
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=((0.0, 1, 1),),
                             ambient_color = ((0.5, 0.5, 0.5),),
                             diffuse_color = ((0.7, 0.7, 0.7),),
                             specular_color = ((0.8, 0.8, 0.8),)
    )


    materials = Materials(
        device=device,
        specular_color=[[1.0, 1.0, 1.0]],
        shininess=65
    )

    if renderer == 'smooth':
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=device, lights=lights)
        )
    elif renderer == 'flat':
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardFlatShader(device=device, lights=lights)
        )
    else:
        raise ValueError("Invalid renderer specification '%s'" % renderer)


    meshes = mesh.extend(batch_size)

    images = renderer(meshes,
                      materials=materials
                      )
    return images


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.util import img_as_ubyte
    from skimage.io import imsave
    from skimage.exposure import rescale_intensity

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    obj_filename = os.path.join(
        r"D:\Workspace\MyRepos\GDL\results\COMA\2020_05_03_23_19_35_Coma\visuals\00004\gt_0000.obj")

    images = render(obj_filename, device, 'smooth')

    images = np.split(images.cpu().numpy(), indices_or_sections=images.shape[0], axis=0)

    # plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        im = img_as_ubyte(rescale_intensity(np.squeeze(images[i]*255), in_range='uint8'))
        # im = np.squeeze(images[i]*255)
        imsave("test_%d.png" % i, im)
        plt.figure()
        plt.imshow(im)
        plt.grid("off")
        plt.axis("off")
        plt.show()
