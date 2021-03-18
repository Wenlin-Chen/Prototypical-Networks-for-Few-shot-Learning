import params_cub as params
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random
import math
from pathlib import Path

"""
Inspired by https://colab.research.google.com/drive/1KzGRSNQpP4BonRKj3ZwGMTGdi-e2y8z-
"""

class CUB(Dataset):
    def __init__(self, files_path, x, y, train=True, transform=True):
      
        self.files_path = files_path
        self.x = x
        self.y = y
        self.train = train
        self.transform = transform
        self.num_files = x.shape[0]
       
    def __len__(self):
        return self.num_files
    
    def __getitem__(self, index):
        file_name = self.x[index][0]
        y = self.y[index][0]
        path = self.files_path/'images'/file_name
        x = read_image(path)
        if self.transform:
            if self.train:
                x = apply_transforms(x)
            else:
                x = center_crop(x)
        else:
            x = cv2.resize(x, (224,224))

        x = normalize(x)
        x =  np.rollaxis(x, 2) # To meet torch's input specification(c*H*W) 
        return x,y

# Helper function for split
def split_class(n_class, n_class_train, n_class_val, mode):
    """
    Return numpy array of disjoint classes for train, val or test set
    """
    np.random.seed(params.seed)
    random_classes = np.random.choice(n_class, n_class, replace=False)

    # Split random class labels into three arrays of class indices
    if mode == 'train':
        return random_classes[:n_class_train]
    elif mode == 'val':
        return random_classes[n_class_train:n_class_val]
    else:
        return random_classes[n_class_val:]


def get_split(class_arr, samples_per_class, images, labels):
    """
    Return images and labels (numpy array) for train, val or test set 
    
    class_arr: random sample of classes for train, val or test set
    samples_per_class: no. of images to take from each class
    images: df of image filenames from images.txt
    labels: df of class labels from image_class_labels.txt
    
    x_train.shape and y_train.shape: (len(class_arr)*samples_per_class, 1)
    indices.shape: (len(class_arr), samples_per_class)
    """
    x, y, indices = None, None, None
    for i in range(len(class_arr)):
        indices_partial = labels.loc[labels['label'] == class_arr[i]].head(
                            samples_per_class).index
        x_partial = images.loc[indices_partial,['name']]
        y_partial = labels.loc[indices_partial,['label']]
        
        # initialise or stack the partial dfs
        if x is None:
            x, y = x_partial, y_partial
            indices = indices_partial.values
            
        else:
            x, y = np.vstack((x, x_partial)), np.vstack((y, y_partial))
            indices = np.vstack((indices, indices_partial.values))
    
    return x, y, indices

# Helper functions for transformation

def apply_transforms(x, sz=(224, 224), zoom=1.05):
    """
    Applies a random crop and horizontal flip
    TO DO: crop 4 corners and center instead of random crop
    """
    sz1 = int(zoom*sz[0])
    sz2 = int(zoom*sz[1])
    x = cv2.resize(x, (sz1, sz2))
    # x = rotate_cv(x, np.random.uniform(-10,10))
    x = random_crop(x, sz[1], sz[0])
    if np.random.rand() >= .5:
                x = np.fliplr(x).copy()
    return x

def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

def random_crop(x, target_r, target_c):
    """ Returns a random crop"""
    r,c,*_ = x.shape
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(rand_r*(r - target_r)).astype(int)
    start_c = np.floor(rand_c*(c - target_c)).astype(int)
    return crop(x, start_r, start_c, target_r, target_c)

def center_crop(im, min_sz=None):
    """ Returns a center crop of an image"""
    r,c,*_ = im.shape
    if min_sz is None: min_sz = min(r,c)
    start_r = math.ceil((r-min_sz)/2)
    start_c = math.ceil((c-min_sz)/2)
    return crop(im, start_r, start_c, min_sz, min_sz)

def normalize(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im/255.0 - imagenet_stats[0])/imagenet_stats[1]

def read_image(path):
    """
    Reads image and converts it from BGR to RGB for visualization
    because opencv uses BGR while matplotlib uses RGB. 
    """
    im = cv2.imread(str(path))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def visualize(dataloader):
    """
    Imshow for Tensor.
    Categories is all the classes and class names
    """
    PATH = Path(params.CUB_data_path)
    classes = pd.read_csv(PATH/"classes.txt", header=None, sep=" ")
    classes.columns = ["id", "class"]
    categories = [x for x in classes["class"]]

    x,y = next(iter(dataloader))
    
    fig = plt.figure(figsize=(10, 10))
    for i in range(8):
      inp = x[i]
      inp = inp.numpy().transpose(1,2,0)
      inp = denormalize(inp)
      
      ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[])
      plt.imshow(inp)
      plt.title(str(categories[y[i]]))

def denormalize(img):
    """Undo normalization in order to visualize"""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return img*imagenet_stats[1] + imagenet_stats[0]