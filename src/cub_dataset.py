import os.path as osp
import PIL
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.data import DataLoader
import torch

import params_cub as params
import pandas as pd

"""
Adapted from https://github.com/Sha-Lab/FEAT/blob/master/model/dataloader/cub.py
"""

class CUB(Dataset):

    def __init__(self, data_folder, mode):
        self.data_folder = data_folder
        images = pd.read_csv(osp.join(self.data_folder,'images.txt'), 
                            header=None, sep=' ')
        images.columns = ['id', 'name']

        labels = pd.read_csv(osp.join(self.data_folder,
                            'image_class_labels.txt'), header=None, sep=' ')
        labels.columns = ['id', 'label']

        class_indices = self.split_class(params.n_class, params.n_class_train,
                        params.n_class_val, mode)
        # print('class array: \n', class_indices)

        self.x, self.y = self.get_split(class_indices, params.samples_per_class, 
                        images, labels)

        self.num_class = np.unique(np.array(self.y)).shape[0]
        print(self.num_class)

        image_size = 224
        if mode == 'train':
            transforms_list = [
                  transforms.TenCrop(params.cropped_size), 
                  transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
                  transforms.Resize(image_size), # GoogLeNet expects img H, W at least 224
                #   transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(256),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # ConvNet-like normalization
        self.transform = transforms.Compose(
            transforms_list + [
            transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                    np.array([0.229, 0.224, 0.225]))
        ])

    # Helper methods for split
    def split_class(self, n_class, n_class_train, n_class_val, mode):
        """
        Return numpy array of disjoint classes for train, val or test set
        """
        np.random.seed(params.seed)
        random_classes = np.random.choice(n_class, n_class, replace=False)

        # Split random class labels into three arrays of class indices
        if mode == 'train':
            return random_classes[:n_class_train]
        elif mode == 'val':
            return random_classes[n_class_train : n_class_train + n_class_val]
        else:
            return random_classes[n_class_train + n_class_val:]


    def get_split(self, class_arr, samples_per_class, images, labels):
        """
        Return images and labels (numpy array) for train, val or test set 
        
        class_arr: random sample of classes for train, val or test set
        samples_per_class: no. of images to take from each class
        images: df of image filenames (string) from images.txt
        labels: df of class labels (integers) from image_class_labels.txt
        
        x.shape and y.shape: (len(class_arr)*samples_per_class, )

        TO DO: export the splits to CSV files
        """
        x, y = None, None
        # indices = None # index of the rows of image or label
        for i in range(len(class_arr)):
            indices_partial = labels.loc[labels['label'] == class_arr[i]].head(
                                samples_per_class).index
            x_partial = images.loc[indices_partial,['name']]
            y_partial = labels.loc[indices_partial,['label']]
            
            # initialise or stack the partial dfs
            if x is None:
                x, y = x_partial, y_partial
                # indices = indices_partial.values
                
            else:
                x, y = np.vstack((x, x_partial)), np.vstack((y, y_partial))
                # indices = np.vstack((indices, indices_partial.values))

        # reshape to array of size 1
        x = x.flatten()
        y = y.flatten()
        # indices = indices.flatten()
        print('flattened x, y: ', x.shape, y.shape)

        return x, y 

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        filename, y = self.x[i], self.y[i]
        file_path = osp.join(self.data_folder, 'images', filename)
        # apply data augmentation transforms
        x = self.transform(Image.open(file_path).convert('RGB'))
        
        return x, y 

def debug(mode):

    dataset = CUB(params.CUB_data_path, mode)
    data_loader = DataLoader(dataset=dataset, batch_size=params.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    for i, batch in enumerate(data_loader):
        print('=========Batch {}==============='.format(i))
        # train: [batch_size, 10, 3, cropped_size, cropped_size], [batch_size]
        # test: [batch_size, 3, 224, 224], [batch_size]
        # if torch.cuda.is_available():
        #     data, label = [_.cuda() for _ in batch]
        #     label = label.type(torch.cuda.LongTensor)
        #     print(data.shape, label.shape) 
        #     return data
        # else:
        data, label = batch
        label = label.type(torch.LongTensor)
        print(data.shape, label.shape) 
        return data

if __name__ == '__main__':
    debug('test')