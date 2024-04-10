import torch
import torch.nn as nn
import kaolin as kal
import kaolin.ops.mesh
from collections import defaultdict, OrderedDict
try:
    import open3d as o3d
except ModuleNotFoundError:
    print("No module named 'open3d'")
from plyfile import PlyData
import copy
try:
    import clip
except ModuleNotFoundError:
    print("No module named 'clip'")
import numpy as np
from torchvision import transforms
from pathlib import Path
from collections import Counter
from modules.Normalization import MeshNormalizer
from modules.mesh import Mesh

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def calc_distances(p0, points):
        return ((p0.to(device) - points.to(device)) ** 2).sum(axis=1)

def fps_from_given_pc(pts, k, given_pc):
    farthest_pts = torch.zeros((k, 3))
    t = given_pc.shape[0] // 3
    farthest_pts[0:t] = given_pc
    indices = [torch.tensor(0).to(device)]
    
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, t):
        distances = torch.min(distances, calc_distances(farthest_pts[i], pts))

    for i in range(t, k):
        farthest_pts[i] = pts[torch.argmax(distances)]
        indices.append(torch.argmax(distances))
        distances = torch.min(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts.to(device), torch.stack(indices).to(device)

def find_red_green_pixels(image):
    # Find pixels where the red channel is largest
    # Find pixels where the green channel is largest
    r_mask = (image[:,:,0] > image[:,:,1]) & (image[:,:,0] > image[:,:,2])
    g_mask = (image[:,:,1] > image[:,:,2]) & (image[:,:,1] > image[:,:,0])
    
    # Get the indices of the red pixels
    r_indices = torch.nonzero(r_mask)
    g_indices = torch.nonzero(g_mask)
    
    return r_indices, g_indices

def show_points(coords, labels, ax, marker_size=87):
    neg_points = coords[labels==0]
    pos_points = coords[labels==1]
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=.5)   
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='lime', marker='o', s=marker_size, edgecolor='white', linewidth=.5)
   

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_anns(anns, plt):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    ax.imshow(img)



def find_ring_neighbors(vertex_index, faces):
    # Masks where the vertex appears in the face
    mask = (faces == vertex_index)
    
    # Extract the faces that contain the vertex
    containing_faces = faces[mask.any(dim=1)]
    
    # Get the neighbors excluding the given vertex
    neighbors = containing_faces[containing_faces != vertex_index]
    
    # Ensure no duplicates
    unique_neighbors = torch.unique(neighbors)
    
    return unique_neighbors


def get_camera_from_view2(elev, azim, r=3.0):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_proj

def get_texture_map_from_color(mesh, color, H=224, W=224):
    num_faces = mesh.faces.shape[0]
    texture_map = torch.zeros(1, H, W, 3).to(device)
    texture_map[:, :, :] = color
    return texture_map.permute(0, 3, 1, 2)


def get_face_attributes_from_color(mesh, color):
    num_faces = mesh.faces.shape[0]
    face_attributes = torch.zeros(1, num_faces, 3, 3).to(device)
    face_attributes[:, :, :] = color
    return face_attributes


def build_ring_dict(faces):
    vertices = torch.unique(faces)
    ring_dict = {}
    for vertex in vertices:
        ring_dict[vertex.item()] = find_ring_neighbors(vertex, faces).tolist()
    
    return ring_dict

def loadmesh(dir, name, load_rings=False):
    
    mesh = Mesh(dir)
    mesh.name = name
    MeshNormalizer(mesh)()
    
    if load_rings:
        rows_for_each_vertex = {}

        for vertex in range(mesh.vertices.shape[0]):
            rows_for_vertex = (mesh.faces == vertex).any(dim=1).nonzero().squeeze().tolist()
            rows_for_each_vertex[vertex] = rows_for_vertex
        
        mesh.rows_for_each_vertex  = rows_for_each_vertex# face idx that contains the vertex

        ring_dict = build_ring_dict(mesh.faces)
        mesh.first_ring = ring_dict

    return mesh


def load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # train on multiple gpus and test on a single gpu. remove 'module' prefix form variable names
        dict_wo_module_prefix = OrderedDict([(".".join(k.split(".")[1:]), v) for k, v in state_dict.items()])
        model.load_state_dict(dict_wo_module_prefix)


# ================== POSITIONAL ENCODERS =============================
class FourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        B = torch.randn((num_input_channels, mapping_size)) * scale
        B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))
        self._B = torch.stack(B_sort)  # for sape

    def forward(self, x):
        # assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels = x.shape

        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        # x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        res = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        # x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        # x = x.permute(0, 3, 1, 2)

        res = 2 * np.pi * res
        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=1)


# mesh coloring helpers
def color_mesh(pred_class, sampled_mesh, colors):
    pred_rgb = segment2rgb(pred_class, colors)
    sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        sampled_mesh.faces)
    MeshNormalizer(sampled_mesh)()

def arbcolor_mesh(sampled_mesh):
    pred_rgb = torch.cat((torch.ones(8036, 1), torch.zeros(8036, 2)), dim=1).to(device)
    sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        sampled_mesh.faces)
    MeshNormalizer(sampled_mesh)()

def setcolor_mesh(sampled_mesh, input_rgb):
    sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
        input_rgb.unsqueeze(0),
        sampled_mesh.faces)
    MeshNormalizer(sampled_mesh)()


def setcolor_mesh_batched(sampled_mesh, input_rgb):
    sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
        input_rgb,
        sampled_mesh.faces)
    MeshNormalizer(sampled_mesh)()

def segment2rgb(pred_class, colors):
    pred_rgb = torch.zeros(pred_class.shape[0], 3).to(device)
    for class_idx, color in enumerate(colors):
        pred_rgb += torch.matmul(pred_class[:,class_idx].unsqueeze(1), color.unsqueeze(0))
        
    return pred_rgb


def add_spheres(mesh, points, colors, vis=False, save_mesh=False, save_path=None, sphere_radius=0.075, sphere_resolution=7, compute_normals=False):
    """
    Parameters
    ----------
    mesh: A 3D mesh. A TriangleMesh object.
    points: The point coordinates of vertices. A numpy array of type np.flot32 with size n x 3.
    colors: The color per point. A numpy array of type np.flot32 with size n x 3. The values should be in the range [0.0, 1.0].
    vis: Whether to visualize the mesh or not. A boolean.
    save_mesh: Whether to save the mesh or not. A boolean.
    save_path: Save path for the file of spheres. A string. The file suffix should be ply.
    sphere_radius: The radius of the sphere. A float.
    sphere_resolution: The mesh resolution of the sphere. An integer.

    Returns
    -------
    spheres: An array of spheres located at the point coordinates. A TriangleMesh object.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=sphere_resolution)

    if compute_normals:
        sphere.compute_vertex_normals()

    pc_data = o3d.geometry.PointCloud()
    pc_data.points = o3d.utility.Vector3dVector(points)

    for i, point_i in enumerate(pc_data.points):
        sphere_i = copy.deepcopy(sphere)

        transformation = np.identity(4)
        transformation[:3, 3] = point_i
        sphere_i.transform(transformation)

        color_i = colors[i]

        sphere_i.paint_uniform_color(color_i)
        mesh += sphere_i

    if vis:
        o3d.visualization.draw_geometries([mesh])
    
    if save_mesh:    
        o3d.io.write_triangle_mesh(save_path, mesh)

    return mesh


def load_ply(file_name):
    ply_data = PlyData.read(file_name)
    vertices = ply_data["vertex"]
    vertices = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    data = {"vertices": vertices}

    faces = np.vstack(ply_data["face"]["vertex_indices"])
    data["faces"] = faces

    try:
        vertex_quality = np.vstack(ply_data["vertex"]["quality"])
        vertex_selection = np.float32(vertex_quality > 0)
        data["vertex_selection"] = vertex_selection
    except ValueError:
        data["vertex_selection"] = None
        print("The ply file %s does not contain quality property for vertex selection." % file_name)

    try:
        face_quality = np.vstack(ply_data["face"]["quality"])
        face_selection = np.float32(face_quality > 0)
        data["face_selection"] = face_selection
    except ValueError:
        data["face_selection"] = None
        print("The ply file %s does not contain quality property for face selection." % file_name)

    return data

def write_ply(file, verts, faces, colors=None):
    with open(file, 'w+') as f:
        # header
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex {}\n".format(verts.shape[0]))
        f.write("property float32 x\nproperty float32 y\nproperty float32 z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("element face {}\n".format(faces.shape[0]))
        f.write("property list uint8 int32 vertex_index\n")
        f.write("end_header\n")

        # vertices
        for vi, v in enumerate(verts):
            if colors is not None:
                f.write("%f %f %f %d %d %d\n" % (v[0], v[1], v[2], colors[vi][0], colors[vi][1], colors[vi][2]))
            else:
                f.write("%f %f %f\n" % (v[0], v[1], v[2]))

        # faces
        for face in faces:
            f.write("3 %d %d %d\n" % (face[0], face[1], face[2]))
