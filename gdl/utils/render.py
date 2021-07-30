import os
import torch
import numpy as np

from gdl.utils.mesh import load_mesh

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

class RendererWrapper(object):

    def __init__(self, renderer, materials, batch_size, device):
        self.renderer = renderer
        self.materials = materials
        self.device = device
        self.batch_size = batch_size

    def render(self, mesh):
        raise NotImplementedError()

    def _prepare_mesh(self, mesh):
        if isinstance(mesh, str):
            # verts, faces, _ = load_obj(obj_filename, load_textures=False, device=device)
            verts, faces, = load_mesh(mesh)
            # faces = faces.verts_idx
        elif isinstance(mesh, list) or isinstance(mesh, tuple):
            verts = mesh[0]
            faces = mesh[1]
            if isinstance(faces, np.ndarray):
                verts = torch.Tensor(verts)
            if isinstance(faces, np.ndarray):
                if faces.dtype == np.uint32:
                    faces = faces.astype(dtype=np.int32)
                faces = torch.Tensor(faces)
        else:
            raise ValueError("Unexpected mesh input of type '%s'. Pass in either a path to a mesh or its vertices "
                             "and faces in a list or tuple" % str(type(mesh)))
        return verts, faces


class ComaMeshRenderer(RendererWrapper):

    def __init__(self, renderer_type, device, image_size=512, num_views=5):
        # Initialize an OpenGL perspective camera.
        # elev = torch.linspace(0, 180, batch_size)
        azim = torch.linspace(-90, 90, num_views)

        R, T = look_at_view_transform(0.35, elev=0, azim=azim,
                                      at=((0, -0.025, 0),), )
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

        # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(device=device, location=((0.0, 1, 1),),
                             ambient_color=((0.5, 0.5, 0.5),),
                             diffuse_color=((0.7, 0.7, 0.7),),
                             specular_color=((0.8, 0.8, 0.8),)
                             )

        materials = Materials(
            device=device,
            specular_color=[[1.0, 1.0, 1.0]],
            shininess=65
        )

        if renderer_type == 'smooth':
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=HardPhongShader(device=device, lights=lights)
            )
        elif renderer_type == 'flat':
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=HardFlatShader(device=device, lights=lights)
            )
        else:
            raise ValueError("Invalid renderer specification '%s'" % renderer_type)

        super().__init__(renderer, materials, num_views, device)

    def render(self, mesh):
        verts, faces = self._prepare_mesh(mesh)

        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)

        verts_rgb[:, :, 0] = 135 / 255
        verts_rgb[:, :, 1] = 206 / 255
        verts_rgb[:, :, 2] = 250 / 255
        #
        # verts_rgb[:,:,0] = 30/255
        # verts_rgb[:,:,1] = 206/255
        # verts_rgb[:,:,2] = 250/255

        # verts_rgb[:,:,0] = 0/255
        # verts_rgb[:,:,1] = 191/255
        # verts_rgb[:,:,2] = 255/255

        textures = Textures(verts_rgb=verts_rgb.to(self.device))
        mesh = Meshes([verts, ], [faces, ], textures)
        mesh = mesh.to(self.device)

        meshes = mesh.extend(self.batch_size)
        images = self.renderer(meshes,
                          materials=self.materials
                          )
        return images


def render(mesh, device, renderer='flat') -> torch.Tensor:
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
    from gdl.utils.image import concatenate_image_batch_to_wide_image, torchFloatToNpUintImage

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    obj_filename = os.path.join(
        # r"D:\Workspace\MyRepos\GDL\data\COMA\template\template.obj")
        r"D:\Workspace\MyRepos\GDL\results\COMA\2020_05_04_15_31_53_Coma\visuals\00001\eval_0000.obj")

    # images = render(obj_filename, device, 'smooth')
    renderer = ComaMeshRenderer('smooth', device, num_views=5)
    images = renderer.render(obj_filename)
    images = concatenate_image_batch_to_wide_image(images)
    # images = images#.cpu()#.numpy()
    # images = images.reshape([-1,] + list(images.shape[2:]))
    # im = img_as_ubyte(rescale_intensity(images.cpu().numpy() * 255, in_range='uint8'))
    im = torchFloatToNpUintImage(images)
    plt.figure()
    plt.imshow(im)
    plt.show()

    # images = np.split(images.cpu().numpy(), indices_or_sections=images.shape[0], axis=0)
    # # plt.figure(figsize=(10, 10))
    # for i in range(len(images)):
    #     im = img_as_ubyte(rescale_intensity(np.squeeze(images[i]*255), in_range='uint8'))
    #     # im = np.squeeze(images[i]*255)
    #     imsave("test_%d.png" % i, im)
    #     plt.figure()
    #     plt.imshow(im)
    #     plt.grid("off")
    #     plt.axis("off")
    #     plt.show()
