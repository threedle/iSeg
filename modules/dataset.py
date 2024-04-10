import torch
from torch.utils.data import Dataset
import os, pickle, cv2
import time
from tqdm import tqdm
from pdb import set_trace
import re
import random
from modules.utils import fps_from_given_pc
from glob import glob


class EncoderDataset(Dataset):
    def __init__(self, data_dir, args):

        _, self.training_inds = fps_from_given_pc(pts=args.mesh.vertices, k=round(args.data_percentage*args.mesh.vertices.shape[0]), given_pc=args.mesh.vertices[0])

        # single click
        random_samf_list = self.prepare_path_list(data_dir, args)

        if args.data_percentage < 1.0:
            random_samf_list = self.get_percentage(random_samf_list, args.data_percentage)
        
        all_data_paths = []
        all_data_paths.extend(random_samf_list)
                
        self.data_paths = all_data_paths
        self.data_length = len(self.data_paths)      
        self.args = args     
    
    def __len__(self):
        return self.data_length

    def __getitem__(self, file_idx):
        data_sample = self._load_data(os.path.join(self.data_paths[file_idx]))
    
        return data_sample
    
    def _load_data(self, file_path):
        
        with torch.no_grad():            
            file = torch.load(file_path)
            sam_f = file['sam_f']
            elev = file['elev'] 
            azim = file['azim']
        
            return sam_f.cuda(), elev.unsqueeze(0).cuda(), azim.unsqueeze(0).cuda() 


    @staticmethod
    def prepare_path_list(data_dir, args):
        save_path = os.path.join(data_dir, 'sam_f', 'encoder_samf_path.pt')
        if not os.path.exists(save_path):
            file_paths = glob(os.path.join(data_dir, 'sam_f', '{}'.format(args.render_res), '*'))
            try:
                torch.save(file_paths, save_path)
            except PermissionError:
                print("Not saving paths list due to permission error")
        else:
            file_paths = torch.load(save_path)

        return file_paths


class DecoderDataset(Dataset):
    def __init__(self, data_dir, args):
        dirs = ['singleclick', 'positive', 'negative']
        fnames = [dir + "_paths_list.pt" for dir in dirs]

        _, self.training_inds = fps_from_given_pc(pts=args.mesh.vertices, k=round(args.data_percentage*args.mesh.vertices.shape[0]), given_pc=args.mesh.vertices[0])

        # single click
        single_paths_list = self.prepare_path_list(data_dir, fnames[0], self.training_inds, 1)
        max_neighbor_samples = len(single_paths_list) * 5

        if args.data_percentage < 1.0:
            single_paths_list = self.get_percentage(single_paths_list, args.data_percentage)
        
        all_data_paths = []
        all_data_paths.extend(single_paths_list)

        # second positive click
        if args.use_positive_click:
            positive_paths_list = self.prepare_path_list(data_dir, fnames[1], self.training_inds, 2)

            if len(positive_paths_list) > max_neighbor_samples:
                positive_paths_list = random.sample(positive_paths_list, max_neighbor_samples)

            if args.data_percentage < 1.0:
                positive_paths_list = self.get_percentage(positive_paths_list, args.data_percentage)
            
            all_data_paths.extend(positive_paths_list)

        # second negative click
        if args.use_negative_click:
            negative_paths_list = self.prepare_path_list(data_dir, fnames[2], self.training_inds, 2)

            if len(negative_paths_list) > max_neighbor_samples:
                negative_paths_list = random.sample(negative_paths_list, max_neighbor_samples)

            if args.data_percentage < 1.0:
                negative_paths_list = self.get_percentage(negative_paths_list, args.data_percentage)

            all_data_paths.extend(negative_paths_list)
                
        self.data_paths = all_data_paths
        self.data_length = len(self.data_paths)  
        self.num_batch = self.data_length/args.batch_size# number of batches per epoch    
        self.args = args     

    @staticmethod
    def get_percentage(paths_list, data_percentage):
        data_len = int(len(paths_list) * data_percentage)
        paths_list_percentage = random.sample(paths_list, data_len)

        return paths_list_percentage
    
    def __len__(self):
        return self.data_length

    def __getitem__(self, file_idx):
        data_sample = self._load_data(os.path.join(self.data_paths[file_idx]))
        return data_sample
    
    def _load_data(self, file_path):
        
        with torch.no_grad():            
            file = torch.load(file_path)
            
            mask0 = file['mask']
            elev, azim = file['viewing_angles']
            selected_vertices = file['selected_vertices']
            input_labels = file['input_label']

            if len(selected_vertices) == 1:
                selected_vertices = selected_vertices.repeat(2)
            
            if len(elev.shape) > 0:
                elev = elev[0]
                azim = azim[0]
                        
            if len(input_labels) == 1:
                input_labels = input_labels.repeat(2)

            # convert labels: 1 to 1 (positive click), 0 to -1 (negative click)
            input_labels = input_labels * 2 - 1
            
            vert_labels = selected_vertices.cuda() * torch.tensor(input_labels).cuda()

            if self.args.return_original == 1:
                # return original image
                image_path = os.path.join(self.args.dataset_path, '{}/{}/target_save_{}_grey.png'.format(self.args.mesh.name, selected_vertices[0], file['view_ind']))
                return mask0.cuda(), vert_labels.cuda(), elev.unsqueeze(0).cuda(), azim.unsqueeze(0).cuda(), image_path
            else:
                return mask0.cuda(), vert_labels.cuda(), elev.unsqueeze(0).cuda(), azim.unsqueeze(0).cuda() 
    
    @staticmethod
    def prepare_path_list(data_dir, fname, training_inds, second_vert_ind):
        save_path = os.path.join(data_dir, fname)
        if not os.path.exists(save_path):
            dir = fname.split("_")[0]
            dir_path = os.path.join(data_dir, dir)

            file_paths = []
            for path, _, files in tqdm(os.walk(dir_path)):
                for name in files:
                    file_paths.append(os.path.join(path, name))
        
            # take only files for selected training vertices and only for mask 0 of SAM
            paths_list = [path for path in file_paths if int(path.split('/')[-1].split('_')[second_vert_ind]) in training_inds and int(path.split('/')[-1].split('_')[-4]) == 0]
            
            try:
                torch.save(paths_list, save_path)
            except PermissionError:
                print("Not saving paths list due to permission error")
        else:
            paths_list = torch.load(save_path)

        return paths_list
