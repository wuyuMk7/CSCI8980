import os, random
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from numpy.lib.function_base import _diff_dispatcher
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
from util import image as img_util

class NoWDataset(Dataset):
    def __init__(self, dataset_path, data_folder, facepos_folder, id_txt, R=6, transform=None):
        self.dataset_path = dataset_path
        self.data_folder = data_folder
        self.facepos_folder = facepos_folder
        self.id_txt = id_txt
        self.R = R
        self.transform = transform

        id_file = os.path.join(dataset_path, id_txt)
        subject_ids = []
        with open(id_file, 'r') as f:
            subject_ids = f.read().split()
        subject_paths = [ 
            os.path.join(dataset_path, data_folder, subject_id) 
            for subject_id in subject_ids
        ]

        all_img_dirs = [ 
            [ os.path.join(subject_path, sub_img_dir) for sub_img_dir in os.listdir(subject_path) ] 
            for subject_path in subject_paths 
        ]
        all_imgs = [
            [ os.path.join(img_path, img) for img_path in img_paths for img in os.listdir(img_path)]
            for img_paths in all_img_dirs
        ]   

        self.subject_paths = subject_paths
        self.all_img_dirs = all_img_dirs
        self.all_imgs = all_imgs

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        # selected_img_paths = [ [ [imgs from the cur person], [img from another person] ] ]
        # selected_img = [ [ imgs from the cur person], [img from another person ] ]
        cur_i = idx
        selected_imgs_paths = []
        random_sample_range = (0, len(self.subject_paths)-1)
        for cur_i in range(len(self.subject_paths)):
            next_i = cur_i
        while len(self.subject_paths) > 1 and next_i == cur_i:
            next_i = random.randint(random_sample_range[0], random_sample_range[1])
        # TODO: should not truncate if R is larger - should add empty images here
        # Now assume R is always valid
        # cur_R = self.R if len(self.all_imgs[cur_i]) > self.R else len(self.all_imgs[cur_i])
        same_img_cnt = self.R-1 if len(self.all_imgs[cur_i]) >= self.R-1 else len(self.all_imgs[cur_i])
        # same_img_paths = random.sample(self.all_imgs[cur_i], cur_R)
        same_img_paths = random.sample(self.all_imgs[cur_i], same_img_cnt)
        dif_img_path = self.all_imgs[next_i][random.randint(0, len(self.all_imgs[next_i])-1)]

        # all_img_contents = [ R * same_imgs, dif_img]
        # all_img_contents = [
        #     [ io.imread(same_img_path) for same_img_path in same_img_paths ],
        #     [ io.imread(dif_img_path) ]
        # ]
        all_img_contents = [ {"image": io.imread(same_img_path), "openpose": np.load(same_img_path.replace(self.data_folder, self.facepos_folder).replace("jpg", "npy"), allow_pickle=True, encoding='latin1')} for same_img_path in same_img_paths]
        all_img_contents.append({"image": io.imread(dif_img_path),"openpose": np.load(dif_img_path.replace(self.data_folder, self.facepos_folder).replace("jpg", "npy"), allow_pickle=True, encoding='latin1')})
        sample = { 'images': all_img_contents }
        
        if self.transform:
            sample = self.transform(sample)
    
        return sample

class ScaleAndCrop(object):
    def __init__(self, config_img_size):
        self.config_img_size = config_img_size
    
    def __call__(self, sample):
        imgs_to_be_processed = sample['images']
        
        imgs_to_be_returned = []
        faceposes_to_be_returned = []
        img_shapes = []
        scales = [] # the 224 / the actual size of img, 2 x 1
        centers = []  # the center of the picture
        for img_facepos in imgs_to_be_processed:
            img = img_facepos['image']
            if np.max(img.shape[:2]) != self.config_img_size:
                # print('Resizing so the max image size is %d..' % self.config_img_size)
                scale = (float(self.config_img_size) / np.max(img.shape[:2]))
            else:
                scale = 1.0#scaling_factor
            center = np.round(np.array(img.shape[:2]) / 2).astype(int)
            # image center in (x,y)
            center = center[::-1]
            crop, proc_param = img_util.scale_and_crop(
                img, scale, center, self.config_img_size)
            # import ipdb; ipdb.set_trace()
            # Normalize image to [-1, 1]
            # plt.imshow(crop/255.0)
            # plt.show()
            crop = 2 * ((crop / 255.) - 0.5)
            single_facepos = img_facepos['openpose']
            # single_facepos *= scale
            faceposes_to_be_returned.append(single_facepos)
            imgs_to_be_returned.append(crop)
            img_shapes.append(np.array([[float(img.shape[0]), 0.],[0., float(img.shape[1])]]))
            scales.append(scale)
            centers.append(center)

        
        return { 'images': imgs_to_be_returned , 'faceposes': faceposes_to_be_returned, 'shape': img_shapes, 'scale': scales, 'centers': centers}

class ToTensor(object):
    def __call__(self, sample):
        imgs_to_be_processed = sample['images']
        faceposes_to_be_processed = np.array(sample['faceposes'])
        img_shapes = np.array(sample['shape'])
        scales = np.array(sample['scale'])
        centers = np.array(sample['centers'])
        
        imgs_to_be_returned = []
        # faceposes_to_be_returned = []
        for img in imgs_to_be_processed:
            new_img = img.transpose((2, 0, 1))
            imgs_to_be_returned.append(new_img)
        # for facepos in faceposes_to_be_processed:
        #     new_facepos = [facepos.item()['top'], facepos.item()['left'], facepos.item()['right'], facepos.item()['bottom']]
        #     faceposes_to_be_returned.append(np.array(new_facepos))
        # np.array(faceposes_to_be_returned)
        faceposes = torch.tensor(faceposes_to_be_processed)
        # print(faceposes.size())
        imgs_to_be_returned = torch.tensor(imgs_to_be_returned)
        img_shapes = torch.tensor(img_shapes)
        scales = torch.tensor(scales)
        centers = torch.tensor(centers)
        # print(imgs_to_be_returned.size())
        return { 'images': imgs_to_be_returned, 'faceposes' :faceposes[:, :, :68, :], 'shapes': img_shapes, 'scales': scales, 'centers':centers}

if __name__ == '__main__':
    config_img_size = 224
    composed_transforms = transforms.Compose([ ScaleAndCrop(config_img_size), ToTensor() ])
    dataset = NoWDataset(
        dataset_path = os.path.join('.', 'training_set', 'NoW_Dataset', 'final_release_version'), 
        data_folder = 'iphone_pictures',
        id_txt = 'subjects_id.txt',
        R = 6,
        transform = composed_transforms
    )

    for i in range(len(dataset)):
        print(dataset[i]['images'].shape)
    plt.imshow(dataset[i]['images'][0].permute(1,2,0).numpy())
    plt.show()