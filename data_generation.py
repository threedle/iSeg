import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys
from modules.utils import device, setcolor_mesh, fps_from_given_pc, show_mask, show_points, loadmesh
from SAM_repo.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from SAM_repo.segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from SAM_repo.segment_anything.utils.transforms import ResizeLongestSide
from modules.render import Renderer, save_renders
import argparse
import re
from tqdm import tqdm
import random


def generate_sam_mask(args):
    # Generate SAM masks conditioned on user clicks for the pre-generated views
   
    # Setting directories
    root = args.decoder_data_dir
    dir = '{}'.format(args.select_vertex)

    if os.path.exists(os.path.join(root, dir, 'save_info.pt')):
        if os.path.isfile(os.path.join(root, dir, 'SAM', 'single_click_{}_mask_{}.png'.format(99, 0))):
            print('vertex', args.select_vertex, 'exist')
            return

        print('vertex', args.select_vertex, 'generating SAM mask')

        save_info = torch.load(os.path.join(root, dir, 'save_info.pt'))

        selectv_2Dcoor_list = save_info['selectv_2Dcoor'] 
        view_ind_list = save_info['view_ind']
        elev_list = save_info['elev_list']
        azim_list = save_info['azim_list']
        
        for i, [[[pointx1, pointy1]], elev, azim] in enumerate(zip(selectv_2Dcoor_list, elev_list, azim_list)):
            
            image = cv2.imread(os.path.join(root, dir, 'target_save_{}_grey.png'.format(i)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            predictor = SamPredictor(sam)
            input_point = np.array([[pointx1.item(), args.render_res-1-pointy1.item()]])
            input_label = np.array([1]) # first single click is always positive
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
                
            # pixel of the selected vertex must be True
            masks[:, args.render_res-1-pointy1.item(), pointx1.item()] = True

            if not os.path.exists(os.path.join(root, 'singleclick')):
                os.makedirs(os.path.join(root, 'singleclick'))

            for mi, (mask, score) in enumerate(zip(masks, scores)):
                if mi > 0:
                    continue

                torch.save({'selected_vertices': torch.tensor([args.select_vertex]), 'input_point': input_point, 'input_label': input_label, 'original_image': image, \
                            'mask': torch.tensor(mask), 'mask_num': mi, 'mask_score': score, 'view_ind': i, 'viewing_angles': (elev, azim)}, \
                                os.path.join(root, 'singleclick', 'vertex_{}_view_{}_masks_{}_model_{}.pt'.format(args.select_vertex, i, mi, args.SAM)))

                
                plt.figure(figsize=(10,10))
                plt.imshow(image)
                show_mask(mask, plt.gca())
                
                show_points(input_point, input_label, plt.gca())
                plt.title(f"Mask {mi+1}, Score: {score:.3f}, Elev: {elev}, Azim: {azim}", fontsize=18)
                plt.axis('off')
                plt.show()  

                os.makedirs(os.path.join(root, dir, 'SAM'), exist_ok=True)
                plt.savefig(os.path.join(root, dir, 'SAM', 'single_click_{}_mask_{}.png'.format(i, mi)))
                plt.close()


def to_pixel_coordinates(face_vertices_image, W, H):
    """
    Convert normalized image coordinates to pixel coordinates.

    Parameters:
    - face_vertices_image: torch.Tensor of shape (..., 2) with normalized coordinates in range [-1, 1]
    - W: Image width
    - H: Image height

    Returns:
    - Pixel coordinates as torch.Tensor of the same shape as face_vertices_image
    """
    return torch.stack([(face_vertices_image[..., 0] + 1) * 0.5 * W,
                        (face_vertices_image[..., 1] + 1) * 0.5 * H], dim=-1)


def generate_save_views_singleclick(args):

    select_vertex = args.select_vertex

    # Setting directories
    root = args.decoder_data_dir

    # Prevent duplicating and overwritting
    if args.overwrite==0 and os.path.isfile(os.path.join(root, '{}/'.format(select_vertex), 'target_save_{}_grey.png'.format(99))):
        print('vertex', select_vertex, 'images, exist')
        return     

    print('vertex', select_vertex, 'generating images')   

    # Set up meshes
    targetmesh = args.mesh  
    targetmesh.face_attributes = targetmesh.face_attributes.float()
    target_rgb = torch.zeros_like(args.mesh.vertices)  
    target_rgb[:] = torch.tensor([2./3., 2./3., 2./3.]).to(device)  
    target_rgb[select_vertex] = torch.tensor([255., 0., 0.]).to(device) # mark the selected vertex red
    setcolor_mesh(targetmesh, target_rgb)
    view_ind_list, selectv_2Dcoor_list, elev_list, azim_list = [], [], [], []


    # Find n_views viewing angles that can see the selected vertex
    for i in range(args.n_views):
        tries = 0
        flag = True
        while True:
            # Elevation: [-pi/2, pi/2]
            elev_rand = -np.pi/2 + torch.rand(1) * np.pi
            
            # Azimuth: [0, 2pi]
            azim_rand = torch.rand(1) * 2 * np.pi

            # if after 200 trials we still can't find a good view
            if tries > 200:
                try:
                    elev_rand = elev_last_work
                    azim_rand = azim_last_work
                except:
                    flag = False
                    break
                tries = 0 # Reset the counting

            # Setting up the renderer
            render_high = Renderer(dim=(args.render_res, args.render_res), 
                                   #lights=torch.tensor([1, random.choice([1., -1.]), 1, random.choice([1., -1.]), 0, 0, 0, 0, 0]),
                                   radius=args.radius)

            # Render
            target_rendered_images, elev, azim, mask_allmesh, vertices_camera, vertices_image_org, face_normals_z = render_high.render_views(targetmesh, num_views=1,
                                                                        show=args.show,
                                                                        random_views = False,
                                                                        center_azim=azim_rand,
                                                                        center_elev=elev_rand,
                                                                        std=args.frontview_std,
                                                                        return_views=True,
                                                                        return_mask = True,
                                                                        return_coordinates=True,
                                                                        lighting=False,
                                                                        background=torch.tensor(args.background).to(device))
            
            image_temp = target_rendered_images.squeeze(0).permute(1, 2, 0)
            vertices_image = to_pixel_coordinates(vertices_image_org, args.render_res-1, args.render_res-1).round().long()

            face_normals_z0 = face_normals_z[0]
            if torch.mean(face_normals_z0[args.mesh.rows_for_each_vertex[select_vertex]]) < 0:
                continue

            x, y = vertices_image[0][select_vertex][0].item(), vertices_image[0][select_vertex][1].detach().cpu().item()
            
            indices_overlaps = torch.where((vertices_image[0][:,0] == x) * (vertices_image[0][:,1] == y))[0]
            vertices_camera_overlaps = vertices_camera[0][torch.where((vertices_image[0][:,0] == x) * (vertices_image[0][:,1] == y))[0]]
            highest_ind = torch.argmax(vertices_camera_overlaps[:, -1])
            rendered_ind = indices_overlaps[highest_ind]
            
            tries += 1
            # if this angle sees the selected vertex
            if rendered_ind == select_vertex and ((target_rendered_images[0][:, args.render_res-1-y, x] - torch.min(target_rendered_images[0][:, args.render_res-1-y, x]))/torch.max((target_rendered_images[0][:, args.render_res-1-y, x] - torch.min(target_rendered_images[0][:, args.render_res-1-y, x]))) == torch.tensor([1, 0, 0]).to(device)).all():
                if mask_allmesh[0][(args.render_res-1-y-2):(args.render_res-1-y+2), (x-2):(x+2)].any() == 0 :
                    # Check if the selected vertex is inside the mask, and not at the edge
                    print('not correct')
                    continue
                
                directory_path = os.path.join(root, '{}/'.format(select_vertex))
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                
                # Remove any colored vertices
                target_rgb[:] = torch.tensor([2./3., 2./3., 2./3.]).to(device)
                setcolor_mesh(targetmesh, target_rgb)
                
                target_rendered_images, elev, azim = render_high.render_views(targetmesh, num_views=1,
                                                                            show=args.show,
                                                                            random_views = False,
                                                                            center_azim=azim_rand,
                                                                            center_elev=elev_rand,
                                                                            std=args.frontview_std,
                                                                            return_views=True,
                                                                            lighting=True,
                                                                            background=torch.tensor(args.background).to(device))
                save_renders(os.path.join(root, '{}/'.format(args.select_vertex)), 0, target_rendered_images, name='target_save_{}_grey.png'.format(i))
                
                # Recolor the vertex
                target_rgb[select_vertex] = torch.tensor([255., 0., 0.]).to(device)
                setcolor_mesh(targetmesh, target_rgb)

                # save the 2D coordinates of the selected vertex
                selectv_2Dcoor_list.append([[x, y]])

                # save the viewing angle index
                view_ind_list.append(i)
                elev_list.append(elev)
                azim_list.append(azim)

                # saved the last worked angel
                elev_last_work = elev
                azim_last_work = azim
                flag=True
                break

        # Cannot find a good view
        if flag==False:
            #print(select_vertex, 'nothing')
            flag=True
            break

    if len(view_ind_list) != 0.:
        # Save the image info (selected vertex, view number 100 in total, 2D coordinates of the selected vertex, and viewing angle)
        combined_arr = {'selected_indices': torch.tensor([select_vertex]), 'view_ind': torch.tensor(view_ind_list), 'selectv_2Dcoor': torch.tensor(selectv_2Dcoor_list), 'elev_list': torch.cat(elev_list), 'azim_list': torch.cat(azim_list)}
        torch.save(combined_arr, os.path.join(root, '{}/save_info.pt'.format(select_vertex)))


def generate_second_negative(args):
    # add a second negative click, in addition to the first positive click

    mesh = args.mesh
    select_vertex = args.select_vertex
    root = args.decoder_data_dir
    dir = '{}'.format(select_vertex)
    
    if os.path.exists(os.path.join(root, dir, 'SAM', 'checkpoint.pt')):
        checkpoint = torch.load(os.path.join(root, dir, 'SAM', 'checkpoint.pt'))
        if 'alldone_negative_vertex_{}_view_{}_savedir'.format(select_vertex, 99) in checkpoint:
            print('all done negative click vertex {}'.format(select_vertex))
            return
    else:
        checkpoint = set()

    # if for some reason (hard to get a good view) the single-click data does not exist, then we skip this vertex
    if not os.path.exists(os.path.join(root, 'singleclick', 'vertex_{}_view_{}_masks_{}_model_{}.pt'.format(select_vertex, 99, 0, args.SAM))):
        print('singleclick data vertex {}'.format(select_vertex), 'skipped')
        return
    
    render_high = Renderer(dim=(args.render_res, args.render_res), 
                                    #lights=torch.tensor([1, random.choice([1., -1.]), 1, random.choice([1., -1.]), 0, 0, 0, 0, 0]),
                                    radius=args.radius)

    save_info = torch.load(os.path.join(root, dir, 'save_info.pt'))
    selectv_2Dcoor_list = save_info['selectv_2Dcoor'] 
    view_ind_list = save_info['view_ind']
    elev_list = save_info['elev_list']
    azim_list = save_info['azim_list']
    targetmesh = mesh  
    target_rgb = torch.zeros_like(mesh.vertices)  
    target_rgb[:] = torch.tensor([2./3., 2./3., 2./3.]).to(device)
    targetmesh.face_attributes = targetmesh.face_attributes.float()
    setcolor_mesh(targetmesh, target_rgb)

    if not os.path.exists(os.path.join(root,'negative')):
        os.makedirs(os.path.join(root, 'negative'))
    
    # Traverse through all the pre-generated single-click images
    for i, [elev, azim] in enumerate(zip(elev_list, azim_list)):
        if 'alldone_negative_vertex_{}_view_{}_savedir'.format(select_vertex, 99) in checkpoint:
            print('all done negative vertex {}, view {}'.format(select_vertex, i))
            continue
        
        # Read the single click masks and other information
        # only read the smallest mask from SAM (the first one)
        save_mask = torch.load(os.path.join(root, 'singleclick', 'vertex_{}_view_{}_masks_{}_model_{}.pt'.format(select_vertex, i, 0, args.SAM)))
        sam_score, sam_mask, input_point, input_label, image  = save_mask['mask_score'],  save_mask['mask'], \
            save_mask['input_point'], save_mask['input_label'], save_mask['original_image']

        # if the single-click mask score is too low
        if sam_score < 0.4:
            continue 
        
        # colored all the selected 3p vertices
        target_rgb[indices_part.long()] = torch.tensor([1., 0, 0]).to(device)
        setcolor_mesh(targetmesh, target_rgb)
        colored_image, elev, azim, mask_allmesh, vertices_camera, vertices_image_org, _ = render_high.render_views(targetmesh, num_views=1,
                                                                    show=args.show,
                                                                    random_views = False,
                                                                    center_azim=azim.unsqueeze(0),
                                                                    center_elev=elev.unsqueeze(0),
                                                                    std=args.frontview_std,
                                                                    return_views=True,
                                                                    return_mask = True,
                                                                    return_coordinates=True,
                                                                    lighting=True,
                                                                    background=torch.tensor(args.background).to(device))
        target_rgb[:] = torch.tensor([2./3., 2./3., 2./3.]).to(device)
        vertices_image = to_pixel_coordinates(vertices_image_org, args.render_res-1, args.render_res-1).round().long()

        # find all the covered vertices
        covered_vertices = {} # key, vertex covered; value (x,y) in sam_masks
        for (x, y) in torch.stack(torch.where(sam_mask == True), dim=-1):
            x, y = x.item(), y.item()
            indices_overlaps = torch.where((vertices_image[0][:,0] == y)*(vertices_image[0][:,1] == args.render_res-1-x))[0]
            if len(indices_overlaps) == 0:
                continue
            vertices_camera_overlaps = vertices_camera[0][indices_overlaps]
            highest_ind = torch.argmax(vertices_camera_overlaps[:, -1])
            rendered_ind_1 = indices_overlaps[highest_ind]
            if rendered_ind_1.item() not in indices_part or (colored_image[0][0, x, y]<= colored_image[0][1, x, y] and colored_image[0][0, x, y]<= colored_image[0][2, x, y]):
                # vertices should be within the 1000 selected ones
                continue
            covered_vertices[rendered_ind_1.item()] = (y, args.render_res-1-x)
        covered_indices = torch.tensor(list(covered_vertices.keys()))

        # if no other vertices within the mask continue
        if len(covered_indices) == 0:
            print('no vertices')
            checkpoint.add('alldone_negative_vertex_{}_view_{}_savedir'.format(select_vertex, i))
            torch.save(checkpoint, os.path.join(root, dir, 'SAM', 'checkpoint.pt'))
            continue
        
        # Calculate the distance of the selected vertex and the covered vertices, only select the close ones
        distance_squared, smallest_indices = ((mesh.vertices[select_vertex] - mesh.vertices[covered_indices]) ** 2).sum(axis=1).sort()
        corresponding_indices_part = covered_indices[smallest_indices[torch.where(distance_squared < 0.5*torch.max(distance_squared))]]
        corresponding_indices_part = corresponding_indices_part[torch.where(corresponding_indices_part != select_vertex)]

        if len(corresponding_indices_part) == 0:
            print('no vertices')
            checkpoint.add('alldone_negative_vertex_{}_view_{}_savedir'.format(select_vertex, i))
            torch.save(checkpoint, os.path.join(root, dir, 'SAM', 'checkpoint.pt'))
            continue
        input_point = input_point.tolist()
        input_label = input_label.tolist()
        # check SAM encoded feature
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        # sample only 5 2nd vertices
        keys = np.array(random.sample(list(corresponding_indices_part), min(5, len(corresponding_indices_part))))
        values = np.array([covered_vertices[key] for key in keys])
        covered_indices_point_coord = np.column_stack((values[:, 0], args.render_res - 1 - values[:, 1])).tolist()
        single_input_points_batched = torch.tensor(input_point*len(covered_indices_point_coord)).to(device)
        double_input_points_batched = torch.stack((single_input_points_batched, torch.tensor(covered_indices_point_coord).to(device)), dim=1).to(device)
        single_input_points_batched = torch.tensor(input_point*len(covered_indices_point_coord)).unsqueeze(1).to(device)
        single_input_labels_batched = torch.tensor(input_label*len(covered_indices_point_coord)).unsqueeze(1).to(device)
        double_input_labels_batched = torch.cat((single_input_labels_batched, torch.zeros_like(single_input_labels_batched)), dim=1).to(device)

        masks, scores, logits = predictor.predict_torch(
            point_coords=predictor.transform.apply_coords_torch(single_input_points_batched, image.shape[:2]),
            point_labels=single_input_labels_batched,
            multimask_output=True
            )
        
        # input masks from previous iterations can make the result better
        mask_input = torch.gather(logits, 1, torch.argmax(scores, dim=1).view(-1, 1, 1, 1).expand(-1, -1, 256, 256))
        masks, scores, logits = predictor.predict_torch(
        point_coords=predictor.transform.apply_coords_torch(double_input_points_batched.to(device), image.shape[:2]),
        point_labels=double_input_labels_batched, # one positive, one negative click
        mask_input=mask_input,
        multimask_output=True
        )
                
        for iter in range(5):
            # input masks from previous iterations can make the result better
            mask_input = torch.gather(logits, 1, torch.argmax(scores, dim=1).view(-1, 1, 1, 1).expand(-1, -1, 256, 256))
            masks, scores, logits = predictor.predict_torch(
            point_coords=predictor.transform.apply_coords_torch(double_input_points_batched, image.shape[:2]),
            point_labels=double_input_labels_batched,
            mask_input=mask_input,
            multimask_output=True
            )
    
        mi = 0
        (indices_org_x, indices_org_y) = torch.where(sam_mask == True)
        for key_i, (key, mask, score) in enumerate(zip(keys, masks[:,mi,:,:], scores[:,mi])):
            # Calculate Intersection and Union
            intersection = torch.sum(sam_mask.cuda() & mask)
            union = torch.sum(sam_mask.cuda() | mask)
            # Compute IoU
            similarity = intersection.float() / union.float()
            if similarity > 0.9 or sam_mask.long().sum()<mask.long().sum():# too similar or mask didn't decrease
                continue
        
            os.makedirs(os.path.join(root, 'negative', str(select_vertex), str(i)), exist_ok=True)
            torch.save({'selected_vertices': torch.tensor([select_vertex, key]), 'input_point': double_input_points_batched[key_i], 'input_label': double_input_labels_batched[key_i], \
                                'mask': mask, 'mask_num': mi, 'mask_score': score, 'view_ind': i, 'viewing_angles': (elev, azim)}, \
            os.path.join(root, 'negative', str(select_vertex), str(i), 'negative_vertices_{}_{}_view_{}_mask_{}_model_{}.pt'.format(select_vertex, key, i, mi, args.SAM)))
        
            if key_i == 0:
                plt.figure(figsize=(10,10))
                plt.imshow(image)
                show_mask(mask.cpu().numpy(), plt.gca())
                show_points(double_input_points_batched[key_i].cpu().numpy(), double_input_labels_batched[key_i].cpu().numpy(), plt.gca(), marker_size=87)
                plt.title(f"Mask {mi+1}, Score: {score:.3f}, Elev: {elev}, Azim: {azim}", fontsize=18)
                plt.axis('off')
                plt.show()  
                plt.savefig(os.path.join(root, dir, 'SAM', 'negative_vertices_{}_{}_view_{}_mask_{}.png'.format(select_vertex, key, i, mi)))
                plt.close()

            input_point, input_label = save_mask['input_point'].tolist(), save_mask['input_label'].tolist()

        checkpoint.add('alldone_negative_vertex_{}_view_{}_savedir'.format(select_vertex, i))
        torch.save(checkpoint, os.path.join(root, dir, 'SAM', 'checkpoint.pt'))
        print('done negative, vertex {}, view {}'.format(select_vertex, i))


def generate_second_positive(args):
    # add a second positive click, in addition to the first positive click

    mesh = args.mesh
    select_vertex = args.select_vertex
    root = args.decoder_data_dir
    dir = '{}'.format(select_vertex)

    # add one positive click, only one positive click previously
    if os.path.exists(os.path.join(root, dir, 'SAM', 'checkpoint.pt')):
        checkpoint = torch.load(os.path.join(root, dir, 'SAM', 'checkpoint.pt'))
        if 'alldone_positive_vertex_{}_view_{}_savedir'.format(select_vertex, 99) in checkpoint:
            print('all done positive click vertex {}'.format(select_vertex))
            return
    else:
        checkpoint = set()

    # if for some reason the single-click data does not exist (hard to get a good view of the click), then we skip this vertex
    if not os.path.exists(os.path.join(root, 'singleclick', 'vertex_{}_view_{}_masks_{}_model_{}.pt'.format(select_vertex, 99, 0, args.SAM))):
        print('singleclick data vertex {}'.format(select_vertex), 'skipped')
        return
    
    render_high = Renderer(dim=(args.render_res, args.render_res), 
                                    #lights=torch.tensor([1, random.choice([1., -1.]), 1, random.choice([1., -1.]), 0, 0, 0, 0, 0]),
                                    radius=args.radius)
    
    save_info = torch.load(os.path.join(root, dir, 'save_info.pt'))
    selectv_2Dcoor_list = save_info['selectv_2Dcoor'] 
    view_ind_list = save_info['view_ind']
    elev_list = save_info['elev_list']
    azim_list = save_info['azim_list']
    targetmesh = mesh  
    target_rgb = torch.zeros_like(mesh.vertices)  
    target_rgb[:] = torch.tensor([2./3., 2./3., 2./3.]).to(device)
    targetmesh.face_attributes = targetmesh.face_attributes.float()
    setcolor_mesh(targetmesh, target_rgb)
    
    # loop over all viewing angles
    for i, [elev, azim] in enumerate(zip(elev_list, azim_list)):
        if 'alldone_positive_vertex_{}_view_{}_savedir'.format(select_vertex, 99) in checkpoint:
            print('all done positive vertex {}, view {}'.format(select_vertex, i))
            continue
        
        # Read the single click masks and other information
        # only read the smallest mask from SAM (the first one)
        save_mask = torch.load(os.path.join(root, 'singleclick', 'vertex_{}_view_{}_masks_{}_model_{}.pt'.format(select_vertex, i, 0, args.SAM)))
        sam_score, sam_mask, input_point, input_label, image  = save_mask['mask_score'],  save_mask['mask'], \
            save_mask['input_point'], save_mask['input_label'], save_mask['original_image']

        # if the original mask score is too low
        if sam_score < 0.4:
            continue 
        
        # colored all the selected 3p vertices
        target_rgb[indices_part.long()] = torch.tensor([1., 0, 0]).to(device)
        setcolor_mesh(targetmesh, target_rgb)
        colored_image, elev, azim, mask_allmesh, vertices_camera, vertices_image_org, _ = render_high.render_views(targetmesh, num_views=1,
                                                                    show=args.show,
                                                                    random_views = False,
                                                                    center_azim=azim.unsqueeze(0),
                                                                    center_elev=elev.unsqueeze(0),
                                                                    std=args.frontview_std,
                                                                    return_views=True,
                                                                    return_mask = True,
                                                                    return_coordinates=True,
                                                                    lighting=True,
                                                                    background=torch.tensor(args.background).to(device))
                                                                    
        target_rgb[:] = torch.tensor([2./3., 2./3., 2./3.]).to(device)
        vertices_image = to_pixel_coordinates(vertices_image_org, args.render_res-1, args.render_res-1).round().long()

        # find all the covered vertices outside the single-click mask
        covered_vertices = {} # key, vertex covered; value (x,y) in sam_masks
        for (x, y) in torch.stack(torch.where((sam_mask.to(device) == False)*((mask_allmesh[0] == 1))), dim=-1):
            x, y = x.item(), y.item()
            indices_overlaps = torch.where((vertices_image[0][:,0] == y)*(vertices_image[0][:,1] == args.render_res-1-x))[0]
            if len(indices_overlaps) == 0:
                continue
            vertices_camera_overlaps = vertices_camera[0][indices_overlaps]
            highest_ind = torch.argmax(vertices_camera_overlaps[:, -1])
            rendered_ind_1 = indices_overlaps[highest_ind]
            if rendered_ind_1.item() not in indices_part or (colored_image[0][0, x, y]<= colored_image[0][1, x, y] and colored_image[0][0, x, y]<= colored_image[0][2, x, y]):
                # vertices should be within the 1000 selected ones
                continue
            covered_vertices[rendered_ind_1.item()] = (y, args.render_res-1-x)

        covered_indices = torch.tensor(list(covered_vertices.keys()))

        # if no other vertices within the mask continue
        if len(covered_indices) == 0:
            print('no vertices')
            checkpoint.add('alldone_positive_vertex_{}_view_{}_savedir'.format(select_vertex, i))
            torch.save(checkpoint, os.path.join(root, dir, 'SAM', 'checkpoint.pt'))
            continue

        # Calculate the distance of the selected vertex and the covered vertices, only select the close ones
        distance_squared, smallest_indices = ((mesh.vertices[select_vertex] - mesh.vertices[covered_indices]) ** 2).sum(axis=1).sort()
        corresponding_indices_part = covered_indices[smallest_indices[torch.where(distance_squared < 0.5*torch.max(distance_squared))]]
        corresponding_indices_part = corresponding_indices_part[torch.where(corresponding_indices_part != select_vertex)]
        
        input_point = input_point.tolist()
        input_label = input_label.tolist()
        
        # check SAM encoded feature
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        keys = np.array(random.sample(list(corresponding_indices_part), min(5, len(corresponding_indices_part))))
        values = np.array([covered_vertices[key] for key in keys])

        # if no sampled covered vertices
        if len(keys) == 0:
            print('no vertices')
            checkpoint.add('alldone_positive_vertex_{}_view_{}_savedir'.format(select_vertex, i))
            torch.save(checkpoint, os.path.join(root, dir, 'SAM', 'checkpoint.pt'))
            continue
        
        covered_indices_point_coord = np.column_stack((values[:, 0], args.render_res - 1 - values[:, 1])).tolist()
        single_input_points_batched = torch.tensor(input_point*len(covered_indices_point_coord))
        double_input_points_batched = torch.stack((single_input_points_batched, torch.tensor(covered_indices_point_coord)), dim=1).to(device)
        single_input_points_batched = torch.tensor(input_point*len(covered_indices_point_coord)).unsqueeze(1)
        single_input_labels_batched = torch.tensor(input_label*len(covered_indices_point_coord)).unsqueeze(1)
        double_input_labels_batched = single_input_labels_batched.repeat(1,2).to(device)

        masks, scores, logits = predictor.predict_torch(
            point_coords=predictor.transform.apply_coords_torch(single_input_points_batched.to(device), image.shape[:2]),
            point_labels=single_input_labels_batched.to(device),
            multimask_output=True
            )
            
        # input masks from previous iterations can make the result better
        mask_input = torch.gather(logits, 1, torch.argmax(scores, dim=1).view(-1, 1, 1, 1).expand(-1, -1, 256, 256))
        masks, scores, logits = predictor.predict_torch(
        point_coords=predictor.transform.apply_coords_torch(double_input_points_batched.to(device), image.shape[:2]),
        point_labels=single_input_labels_batched.repeat(1,2).to(device), 
        mask_input=mask_input,
        multimask_output=True
        )
            
        for iter in range(5):
            # input masks from previous iterations can make the result better
            mask_input = torch.gather(logits, 1, torch.argmax(scores, dim=1).view(-1, 1, 1, 1).expand(-1, -1, 256, 256))
            masks, scores, logits = predictor.predict_torch(
            point_coords=predictor.transform.apply_coords_torch(double_input_points_batched.to(device), image.shape[:2]),
            point_labels=single_input_labels_batched.repeat(1,2).to(device),
            mask_input=mask_input,
            multimask_output=True,
        )

        mi = 0
        
        # all data is saved in 'positive folder'
        if not os.path.exists(os.path.join(root,'positive')):
            os.makedirs(os.path.join(root, 'positive'))
        
        for key_i, (key, mask, score) in enumerate(zip(keys, masks[:,mi,:,:], scores[:,mi])):
            # Calculate Intersection and Union
            intersection = torch.sum(sam_mask.cuda() & mask)
            union = torch.sum(sam_mask.cuda() | mask)
            
            # Compute IoU
            similarity = intersection.float() / union.float()
            if similarity > 0.9 or sam_mask.long().sum()>mask.long().sum():
                # do not include masks that are too similar to the original one or mask didn't increase
                continue
            
            os.makedirs(os.path.join(root, 'positive', str(select_vertex), str(i)), exist_ok=True)
            torch.save({'selected_vertices': torch.tensor([select_vertex, key]), 'input_point': double_input_points_batched[key_i], 'input_label': double_input_labels_batched[key_i], \
                                'mask': mask, 'mask_num': mi, 'mask_score': score, 'view_ind': i, 'viewing_angles': (elev[0], azim[0])}, \
            os.path.join(root, 'positive', str(select_vertex), str(i), 'positive_vertices_{}_{}_view_{}_mask_{}_model_{}.pt'.format(select_vertex, key, i, mi, args.SAM)))
            
            if key_i  == 0:
                plt.figure(figsize=(10,10))
                plt.imshow(image)
                show_mask(mask.cpu().numpy(), plt.gca())
                show_points(double_input_points_batched[key_i].cpu().numpy(), single_input_labels_batched.repeat(1,2)[key_i].cpu().numpy(), plt.gca(), marker_size=87)
                plt.title(f"Mask {mi+1}, Score: {score:.3f}, Elev: {elev}, Azim: {azim}", fontsize=18)
                plt.axis('off')
                plt.show()  
                plt.savefig(os.path.join(root, dir, 'SAM', 'positive_vertices_{}_{}_view_{}_mask_{}.png'.format(select_vertex, key, i, mi)))
                plt.close()
            
        checkpoint.add('alldone_positive_vertex_{}_view_{}_savedir'.format(select_vertex, i))
        torch.save(checkpoint, os.path.join(root, dir, 'SAM', 'checkpoint.pt'))
        print('done positive, vertex {}, view {}'.format(select_vertex, i))
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--seed', type=int, default=0)

    # data generation parameters
    parser.add_argument('--SAM', type=str, default='vit_h') # select SAM model, default is huge (we only have ViT_huge model)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--percentage', type=float, default=0.03) # percentage of the vertices
    parser.add_argument('--single_click', type=int, default=1) # generate single click data
    parser.add_argument('--second_positive', type=int, default=0) # generate second positive click data
    parser.add_argument('--second_negative', type=int, default=0) # generate second negative click data
    parser.add_argument('--overwrite', type=int, default=0)

    # Paths and names
    parser.add_argument('--obj_path', type=str, default='./meshes/hammer.obj') # directory of the mesh object
    parser.add_argument('--name', type=str, default='hammer') # mesh name
    parser.add_argument('--decoder_data_dir', type=str, default='./data/hammer/decoder_data') # directory to store the generated 2D data

    # render parameters
    parser.add_argument('--background', nargs=3, type=float, default=[1., 1., 1.])
    parser.add_argument('--n_views', type=int, default=100)
    parser.add_argument('--frontview_std', type=float, default=4)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.])
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--render_res', type=int, default=224) # render resolution

    args = parser.parse_args()

    # Load SAM model
    sam_checkpoint = os.path.join('./SAM_repo/model_checkpoints/', "sam_vit_h_4b8939.pth")
    model_type = args.SAM
    device = 'cuda' and torch.cuda.is_available() or 'cpu'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Load mesh object
    args.mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)
    
    # Create path for data storage
    if not os.path.exists(args.decoder_data_dir):
        os.makedirs(args.decoder_data_dir, exist_ok=True)

    # Select a part of the vertices uniformly
    vertices_part, indices_part = fps_from_given_pc(pts=args.mesh.vertices, k=round(args.mesh.vertices.shape[0]*args.percentage), given_pc=args.mesh.vertices[0])

    for ind in tqdm(range(len(indices_part))):
        args.select_vertex = indices_part[ind].item()
        if args.single_click == 1:
            # generate views
            generate_save_views_singleclick(args)
            
            # generate SAM masks
            generate_sam_mask(args)

        if args.second_positive == 1:
            generate_second_positive(args)
            
        if args.second_negative == 1:
            generate_second_negative(args)
