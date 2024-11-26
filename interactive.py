import argparse
import numpy as np
import open3d as o3d
import os
import copy
import random
from tqdm import tqdm
import time
from skimage import filters
import polyscope as ps
import polyscope.imgui as psim
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from modules.utils import device, setcolor_mesh_batched, load_state_dict, find_red_green_pixels, show_mask, show_points, loadmesh, add_spheres, write_ply
from modules.dataset import DecoderDataset
from modules.render import save_renders, Renderer
from modules.click_attention import ClickAttention
from modules.decoder import Decoder


# Append to the existing files or create them if they don't exist
def save_or_append(filename, data):
    if os.path.exists(filename):
        existing_data = torch.load(filename, map_location=device)
        if len(data.shape)<len(existing_data.shape):
            combined_data = torch.cat([existing_data, data.unsqueeze(0)])
        else:
            combined_data = torch.cat([existing_data, data])
        torch.save(combined_data, filename)
    else:
        torch.save(data, filename)


def save_loss(loss, dir, name=None):
    plt.figure()
    plt.plot(loss)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    # Save the figure
    if name is not None:
        plt.title(name+' over time')
        plt.savefig(os.path.join(dir, name+'.jpg'))
        plt.close()
    else:
        plt.title('Loss over time')
        plt.savefig(os.path.join(dir, 'loss.jpg'))
        plt.close()


def load_features(args):
    # load features
    if torch.cuda.is_available():
        pred_f = torch.load(args.encoder_f_path)
    else:
        pred_f = torch.load(args.encoder_f_path, map_location=torch.device('cpu'))
    return pred_f


def masked_bce_loss(y_pred, y_true, mask):
    # Compute the raw BCE loss term-wise
    bce = F.binary_cross_entropy(y_pred, F.one_hot(y_true.long(), num_classes=2).squeeze(1).permute(0, 3, 1, 2).float(), reduction='none')
    mask = torch.cat((mask.unsqueeze(1), mask.unsqueeze(1)), dim=1)
    # Apply the mask
    masked_bce = bce * mask
    
    # Compute the mean of the masked BCE values
    loss = masked_bce.sum() / mask.sum()
    return loss

def load_model(args):
    print('load model')
    
    num_gpus = torch.cuda.device_count()
    print('number of avilable gpus: %d' % num_gpus)

    # read learned 3D features
    pred_f = load_features(args)

    # MLP Settings: predicting vertex segmentation probability
    attention = ClickAttention(256, args).to(device)
    mlp = Decoder(depth=args.depth, width=[512]+[256]*args.depth, out_dim=2, input_dim=512, positional_encoding=args.positional_encoding,
                            sigma=args.sigma).to(device)
        
    save_path = os.path.join(args.save_dir, args.model_name)
    print('decoder checkpoint path: %s' % save_path)
    
    checkpoint = torch.load(save_path, map_location=device)
    load_state_dict(attention, checkpoint['attention_state_dict'])
    load_state_dict(mlp, checkpoint['model_state_dict'])
    
    return pred_f, attention, mlp


def run_model(args, pred_f, attention, mlp, test_index):
    if len(test_index.shape) == 1:
        test_index = test_index.unsqueeze(0)
    
    batch_size = test_index.shape[0]
    
    feature_field_batch = pred_f.unsqueeze(0).expand(batch_size, -1, -1)
    start = time.time()
    
    # click attentntion
    weighted_vals = attention(feature_field_batch, test_index)
        
    input_tensor = torch.cat((feature_field_batch, weighted_vals), dim=-1)
            
    # mlp
    prob_tensor = mlp(input_tensor)
    end = time.time()
    print('inference time: %.2f seconds' % (end - start))

    # vertex probability
    prob_tensor_np = prob_tensor.detach().cpu().numpy().astype(np.float32)
    seg_prob = prob_tensor_np[0, :, 1]
    
    return seg_prob
    

def prepare_vertex_color(prob_tensor, base_color, seg_color):
    base_color = np.array([base_color], dtype=np.float32) / 255.
    seg_color = np.array([seg_color], dtype=np.float32) / 255.

    vertex_color = prob_tensor.T * seg_color + (1. - prob_tensor.T) * base_color
    
    return vertex_color


def get_user_selection(vertices, faces):
    vlen = len(vertices)
    flen = len(faces)

    structure, idx = ps.get_selection()
    if idx < vlen:
        sel_type = 'vertex'
        vert_idx = idx
    elif idx < vlen + flen:
        sel_type = 'face'
        index = idx - vlen

        face_verts = faces[index]
        vert_idx = face_verts[0]
    else:
        sel_type = 'edge'
        index = idx - vlen - flen
        vert_idx = 'None'

    return sel_type, vert_idx


def update_segmentation(args, pred_f, attention, mlp, selected_vertices, use_otsu_is_true):  
    prob_soft = run_model(args, pred_f, attention, mlp, test_index=torch.tensor(selected_vertices))

    if use_otsu_is_true:
        otsu_thresh = filters.threshold_otsu(prob_soft)
        print("Otsu treshold: %.3f" % otsu_thresh)

        hard_prob = np.array(prob_soft >= otsu_thresh).astype(prob_soft.dtype)
        seg_prob = hard_prob
    else:
        seg_prob = prob_soft
    
    vertex_color = prepare_vertex_color(np.expand_dims(seg_prob, 0), args.base_color, args.seg_color)

    ps.get_surface_mesh("mesh").add_color_quantity("segmentation color", vertex_color, enabled=True)


def callback():
    global click_modes, selected_click_mode, click_types, selected_click_type, use_otsu_is_true
    global vertices, faces, args, pred_f, attention, mlp
    global sel_type, selected_vertices, last_selected_vertex
    global ps_curve

    # GUI
    psim.TextUnformatted("iSeg Interactive Module")

    psim.Separator()

    # Combo box to choose from options
    # There, the options are a list of strings in `click_modes`,
    # and the currently selected element is stored in `selected_click_mode`.
    psim.PushItemWidth(200)
    changed = psim.BeginCombo("Choose click mode", selected_click_mode)
    if changed:
        for val in click_modes:
            _, selected = psim.Selectable(val, selected_click_mode==val)
            if selected:
                selected_click_mode = val
        psim.EndCombo()
    psim.PopItemWidth()

    # Combo box to choose from options
    # There, the options are a list of strings in `click_types`,
    # and the currently selected element is stored in `selected_click_type`.
    psim.PushItemWidth(200)
    changed = psim.BeginCombo("Choose click type", selected_click_type)
    if changed:
        for val in click_types:
            _, selected = psim.Selectable(val, selected_click_type==val)
            if selected:
                selected_click_type = val
        psim.EndCombo()
    psim.PopItemWidth()

    selected_vertices_str = '[' + ' '.join(['%d' % idx for idx in selected_vertices]) + ']'
    psim.TextUnformatted(f"Selected vertices: {selected_vertices_str}")
    
    if(psim.Button("Undo")):
        # This code is executed when the button is pressed      
        if len(selected_vertices) > 0:
            last_selected_vertex = selected_vertices[-1]
            selected_vertices = selected_vertices[:-1]

            if len(selected_vertices) > 0:
                update_segmentation(args, pred_f, attention, mlp, selected_vertices, use_otsu_is_true)
            else:
                ps_colors = np.ones_like(vertices) * np.array([args.base_color], dtype=np.float32) / 255.0
                ps.get_surface_mesh("mesh").add_color_quantity("segmentation color", ps_colors, enabled=True)
            
    # By default, each element goes on a new line. Use this 
    # to put the next element on the _same_ line.
    psim.SameLine()

    if(psim.Button("Redo")):
        # This code is executed when the button is pressed
        not_first_negative = len(selected_vertices) == 0 and last_selected_vertex > 0
        different_from_last = len(selected_vertices) > 0 and last_selected_vertex != selected_vertices[-1]

        if last_selected_vertex is not None:
            if not_first_negative or different_from_last:
                selected_vertices.append(last_selected_vertex)
                
                update_segmentation(args, pred_f, attention, mlp, selected_vertices, use_otsu_is_true)

    # By default, each element goes on a new line. Use this 
    # to put the next element on the _same_ line.
    psim.SameLine()
    
    if(psim.Button("Reset")):
        # This code is executed when the button is pressed
        if len(selected_vertices) > 0:
            last_selected_vertex = selected_vertices[-1]    
        
        selected_vertices = []

        ps_colors = np.ones_like(vertices) * np.array([args.base_color], dtype=np.float32) / 255.0
        ps.get_surface_mesh("mesh").add_color_quantity("segmentation color", ps_colors, enabled=True)
    
    # Otsu threshold flag
    use_otsu_changed, use_otsu_is_true = psim.Checkbox("Use Otsu Threshold", use_otsu_is_true) 
    if use_otsu_changed:
        if len(selected_vertices) > 0:
            update_segmentation(args, pred_f, attention, mlp, selected_vertices, use_otsu_is_true)
    
    if selected_click_mode != click_modes[0]:
        # get current user selection
        sel_type, selected_vertex = get_user_selection(vertices, faces)

        valid_sel = (sel_type == 'vertex' or sel_type == 'face') and selected_vertex != 0
        if last_selected_vertex is not None:
            valid_sel = valid_sel and selected_vertex != np.abs(last_selected_vertex)
        
        different_from_last = len(selected_vertices) > 0 and selected_vertex != np.abs(selected_vertices[-1])
        
        # single click mode
        if selected_click_mode == click_modes[1]:
            if valid_sel:
                if len(selected_vertices) == 0 or different_from_last:
                    selected_vertices = [selected_vertex]
                    update_segmentation(args, pred_f, attention, mlp, selected_vertices, use_otsu_is_true)
        # multiple clicks mode
        else:
            if valid_sel:
                if selected_click_type == click_types[0]: # positive click
                    if len(selected_vertices) == 0 or different_from_last:
                        selected_vertices.append(selected_vertex)
                        update_segmentation(args, pred_f, attention, mlp, selected_vertices, use_otsu_is_true)
                else: # negative click
                    if different_from_last:
                        selected_vertices.append(-selected_vertex)
                        update_segmentation(args, pred_f, attention, mlp, selected_vertices, use_otsu_is_true)
    
    # show clicked points
    click_colors = []
    click_poses = []
    if len(selected_vertices) > 0:
        for selected_vertex in selected_vertices:
            click_poses.append(np.expand_dims(vertices[np.abs(selected_vertex)], axis=0))

            if selected_vertex >= 0:
                click_colors.append(np.expand_dims(args.pos_color, axis=0))
            else:
                click_colors.append(np.expand_dims(args.neg_color, axis=0))

        click_poses_np = np.concatenate(click_poses, axis=0)
        click_colors_np = np.concatenate(click_colors, axis=0)
        edges_np = np.array([[i,i] for i in range(len(selected_vertices))])

        ps_curve = ps.register_curve_network("clicked points", click_poses_np, edges_np, radius=0.01, enabled=True)
        ps_curve.add_color_quantity("click color", click_colors_np, enabled=True)
    else:
        if ps_curve is not None:
            ps_curve.set_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # directory structure
    parser.add_argument('--obj_path', type=str, default='./meshes/hammer.obj')
    parser.add_argument('--encoder_f_path', type=str, default='./demo/hammer/encoder/pred_f.pth')
    parser.add_argument('--decoder_data_dir', type=str, default='./data/hammer/decoder_data')
    parser.add_argument('--save_dir', type=str, default='./demo/hammer/decoder/')
    parser.add_argument('--model_name', type=str, default='decoder_checkpoint.pth')

    # mesh + data info
    parser.add_argument('--name', type=str, default='hammer')
    parser.add_argument('--data_percentage', type=float, default=1.0)
    parser.add_argument('--views_per_vert', type=int, default=100)

    # render
    parser.add_argument('--background', nargs=3, type=float, default=[1., 1., 1.])
    parser.add_argument('--n_views', type=int, default=1)
    parser.add_argument('--frontview_std', type=float, default=4)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[3.14, 0.])
    parser.add_argument('--render_res', type=int, default=224)

    # attention
    parser.add_argument('--use_attention_q', type=int, default=1)
    parser.add_argument('--use_attention_k', type=int, default=1)
    parser.add_argument('--use_attention_v', type=int, default=1)
    parser.add_argument('--redsidual_attention', type=int, default=0)
    parser.add_argument('--scale_attention', type=int, default=1)
    
    # network
    parser.add_argument('--continue_train', type=int, default=0)
    parser.add_argument('--depth', type=int, default=14)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=256) # 256 channels for SAM embedding feature
    parser.add_argument('--positional_encoding', action='store_true')
    parser.add_argument('--sigma', type=float, default=5.0)

    # optimization
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--return_original', type=int, default=0)
    
    # parallel, multi-GPU training
    parser.add_argument('--use_data_parallel', type=int, default=0)
    
    # mode
    parser.add_argument('--select_vertices', nargs='+', type=int, default=[0])
    parser.add_argument('--show', type=int, default=0)

    # visualization
    parser.add_argument('--base_color', nargs=3, type=int, default=[180, 180, 180])
    parser.add_argument('--show_seg', type=int, default=1)
    parser.add_argument('--seg_color', nargs=3, type=int, default=[28, 99, 227])

    parser.add_argument('--show_spheres', type=int, default=0)
    parser.add_argument('--sphere_radius', type=float, default=0.025)
    parser.add_argument('--pos_color', nargs=3, type=int, default=[0, 255, 0])
    parser.add_argument('--neg_color', nargs=3, type=int, default=[255, 0, 0])

    args = parser.parse_args()
    
    # GUI config
    click_modes = ["Disable", "Single Click", "Multiple Clicks"]
    selected_click_mode = click_modes[1]
    click_types = ["Positive", "Negative"]
    selected_click_type = click_types[0]
    use_otsu_is_true = False
    selected_vertices = []
    last_selected_vertex = None
    
    # Load mesh object
    args.mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=False)
    vertices = args.mesh.vertices.detach().cpu().numpy()
    faces = args.mesh.faces.detach().cpu().numpy()

    # create decoder directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # load model        
    pred_f, attention, mlp = load_model(args)

    # initi polyscope
    ps.init()
    
    # by default, Polyscope caches all the viz data from the previous run 
    ps.remove_all_structures()
    ps.set_navigation_style("free")
    ps_mesh = ps.register_surface_mesh("mesh", vertices, faces, smooth_shade=True)

    vertex_color = np.ones_like(vertices) * np.array([args.base_color], dtype=np.float32) / 255.0
    ps.get_surface_mesh("mesh").add_color_quantity("segmentation color", vertex_color, enabled=True)

    ps_curve = None

    ps.set_invoke_user_callback_for_nested_show(True)
    ps.set_user_callback(callback)
    ps.show()
