from modules.mesh import Mesh
import kaolin as kal
from modules.utils import get_camera_from_view2
import matplotlib.pyplot as plt
from modules.utils import device
import torch
import numpy as np
import torchvision
import os


def save_renders(dir, i, rendered_images, name=None):
    if name is not None:
        torchvision.utils.save_image(rendered_images, os.path.join(dir, name))
    else:
        torchvision.utils.save_image(rendered_images, os.path.join(dir, 'renders/iter_{}.jpg'.format(i)))



class Renderer():

    def __init__(self, mesh='sample.obj',
                 lights=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 camera=kal.render.camera.generate_perspective_projection(np.pi / 3).to(device),
                 dim=(224, 224), radius=2.):

        if camera is None:
            camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)

        self.lights = lights.unsqueeze(0).to(device)
        self.camera_projection = camera
        self.dim = dim
        self.radius = radius

    def render_views(self, mesh, num_views=8, std=8, random_views = False, center_elev=0, center_azim=0, show=False, lighting=True,
                           background=None, mask=False, return_views=False, return_mask=False, return_features=False, return_coordinates=False):
        # Front view with small perturbations in viewing angle
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        if random_views:
            elev = torch.randn(num_views) * np.pi / std + center_elev
            azim = torch.randn(num_views) * 2 * np.pi / std + center_azim
        else:
            elev = center_elev#torch.tensor([0.3741]) * np.pi / std + center_elev
            azim = center_azim#torch.tensor([0.3741]) * 2 * np.pi / std + center_azim

        images = []
        masks = []
        rgb_mask = []
        vertices_cameras = []
        vertices_images = []
        face_normals_zs = []

        if background is not None:
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=self.radius).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                camera_transform=camera_transform)
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            vertices_camera, vertices_image, face_normals = \
            self.prepare_vertices_ours(mesh.vertices.to(device), mesh.faces.to(device), camera_proj=self.camera_projection, camera_rot=None, camera_trans=None, camera_transform=camera_transform)
            face_normals_z = face_normals[:, :, -1]
            masks.append(soft_mask)
            vertices_cameras.append(vertices_camera)
            vertices_images.append(vertices_image)
            face_normals_zs.append(face_normals_z)

            

            if background is not None:
                image_features, mask = image_features

            if return_features:
                image = image_features
            else:
                image = torch.clamp(image_features, 0.0, 1.0)
                

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, image.shape[3], 1, 1).permute(0, 2, 3, 1).to(device)
                if return_features:
                    image = image_features
                else:
                    image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                background_idx = torch.where(mask == 0)
                assert torch.all(image[background_idx] == torch.zeros(image.shape[3]).to(device))
                background_mask[background_idx] = background#.repeat(background_idx[0].shape)
                if return_features:
                    image = image + background_mask
                else:
                    image = torch.clamp(image + background_mask, 0., 1.)
                
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.show()

        # Start with always returning images
        result = (images,)

        # If return_views is True, add elev and azim
        if return_views:
            result += (elev, azim)

        # If return_mask is True, add masks
        if return_mask:
            result += (masks,)

        # If return_coordinates is True, add face vertices
        if return_coordinates:
            result += (torch.cat(vertices_cameras), torch.cat(vertices_images), torch.cat(face_normals_zs))

        return result
        
    def get_camera_from_view_batched(self, elev, azim, r=2.0, look_at_height=0.0, device=torch.device('cuda')):
        """
        Convert tensor elevation/azimuth values into camera projections 

        Args:
            elev (torch.Tensor): elevation
            azim (torch.Tensor): azimuth
            r (float, optional): radius. Defaults to 3.0.

        Returns:
            Camera projection matrix (B x 4 x 3)
        """
        x = r * torch.cos(elev) * torch.cos(azim)
        y = r * torch.sin(elev)
        z = r * torch.cos(elev) * torch.sin(azim)
        # print(elev,azim,x,y,z)
        B = elev.shape[0]

        if len(x.shape) == 0:
            pos = torch.tensor([x,y,z]).unsqueeze(0).to(device)
        else:
            pos = torch.stack([x, y, z], dim=1)

        # look_at = -pos
        look_at = torch.zeros_like(pos)
        look_at[:, 1] = look_at_height

        up = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).repeat(B, 1).to(device)

        if len(pos.shape) == 4:
            pos = pos.squeeze(-1)
            pos = pos.squeeze(-1)
            look_at = look_at.squeeze(-1)
            look_at = look_at.squeeze(-1)

        if len(pos.shape) == 3:
            pos = pos.squeeze(2)
            look_at = look_at.squeeze(2)
        
        camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, up).to(device)
        return camera_proj
    

    def prepare_vertices_ours(self, vertices, faces, camera_proj, camera_rot=None, camera_trans=None,
                     camera_transform=None):
        
        # Apply the transformation from camera_rot and camera_trans or camera_transform
        if camera_transform is None:
            assert camera_trans is not None and camera_rot is not None, \
                "camera_transform or camera_trans and camera_rot must be defined"
            vertices_camera = kal.render.camera.rotate_translate_points(vertices, camera_rot,
                                                            camera_trans)
        else:
            assert camera_trans is None and camera_rot is None, \
                "camera_trans and camera_rot must be None when camera_transform is defined"
            padded_vertices = torch.nn.functional.pad(
                vertices, (0, 1), mode='constant', value=1.
            )
            vertices_camera = (padded_vertices @ camera_transform)
        # Project the vertices on the camera image plan
        vertices_image = kal.render.camera.perspective_camera(vertices_camera, camera_proj)
        face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
        face_normals = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)
        #vertex_normals = kal.ops.mesh.compute_vertex_normals(faces, face_normals, num_vertices=vertices.shape)
        return vertices_camera, vertices_image, face_normals
    
    def render_views_batched(self, mesh, num_views=8, std=8, random_views = False, center_elev=0, center_azim=0, show=False, lighting=True,
                           background=None, needs_repeat = False, mask=False, return_views=False, return_mask=False, return_features=False, return_coordinates=False):
        # Front view with small perturbations in viewing angle
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]
        #colors = mesh.face_attributes

        if random_views:
            elev = torch.randn(num_views) * np.pi / std + center_elev
            azim = torch.randn(num_views) * 2 * np.pi / std + center_azim
        else:
            elev = center_elev#torch.tensor([0.3741]) * np.pi / std + center_elev
            azim = center_azim#torch.tensor([0.3741]) * 2 * np.pi / std + center_azim

        if needs_repeat:
            if background is not None:
                face_attributes = [mesh.face_attributes.repeat(num_views,1,1,1),
                    torch.ones((num_views, n_faces, 3, 1), device=device)]
            else:
                face_attributes = (mesh.face_attributes).repeat(num_views,1,1,1)
        else:
            if background is not None:
                face_attributes = [mesh.face_attributes,
                    torch.ones((num_views, n_faces, 3, 1), device=device)
                ]
            else:
                face_attributes = mesh.face_attributes

        # A(r), B(b)
        # 1,2,3,4 
        # 
        # 1, AB(r,b), 2 AB(r,g)
        # (r,b,r,g,a,) 
        camera_transform = self.get_camera_from_view_batched(elev.to(device), azim.to(device))
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                    mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                    camera_transform=camera_transform)
        
        vertices_camera, vertices_image, face_normals = \
            self.prepare_vertices_ours(mesh.vertices.to(device), mesh.faces.to(device), camera_proj=self.camera_projection, camera_rot=None, camera_trans=None, camera_transform=camera_transform)
        face_normals_z = face_normals[:, :, -1]
        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals_z)
       

        if background is not None:
            image_features, mask = image_features

        if return_features:
            image = image_features
        else:
            image = torch.clamp(image_features, 0.0, 1.0)
       

        if lighting:
            #self.lights = self.lights.repeat(num_views, 1, 1)
            image_normals = face_normals[:, face_idx].squeeze(0)
            image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
            image = image * image_lighting.repeat(1, image.shape[3], 1, 1).permute(0, 2, 3, 1).to(device)
            if return_features:
                image = image_features
            else:
                image = torch.clamp(image_features, 0.0, 1.0)

        if background is not None:
            background_mask = torch.zeros(image.shape).to(device)
            mask = mask.squeeze(-1)
            background_idx = torch.where(mask == 0)
            assert torch.all(image[background_idx] == torch.zeros(image.shape[3]).to(device))
            background_mask[background_idx] = background#.repeat(background_idx[0].shape)
            if return_features:
                image = image + background_mask
            else:
                image = torch.clamp(image + background_mask, 0., 1.)

        images = image.permute(0, 3, 1, 2)
        masks = soft_mask


        # Start with always returning images
        result = (images,)

        # If return_views is True, add elev and azim
        if return_views:
            result += (elev, azim)

        # If return_mask is True, add masks
        if return_mask:
            result += (masks,)

        # If return_coordinates is True, add face vertices
        if return_coordinates:
            result += (vertices_camera, vertices_image, face_normals_z)

        return result

    
    