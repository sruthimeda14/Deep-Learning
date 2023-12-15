import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import random_split
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import *
from dataloader import TTDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import cv2
from PIL import Image
from copy import deepcopy
from pprint import pprint

load_model_path = '/checkpoints'
data_path = '/media/E/dataset/dataset'
output_dir = 'output_images_cropped'
batch_size = 1
img_x, img_y = 320, 128
learning_rate = 1e-4
dropout = 0.5

use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None


def un_normalize(pos_xy):

    return [pos_xy[0]*1920, pos_xy[1]*1080]

def get_crop_coords(coord, width, height, crop_size):
    """
    Calculate the crop coordinates around a single coordinate.

    Args:
        coord (tuple): Tuple of (x, y) coordinates of the center of the crop region.
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.
        crop_size (int): Size of the square crop region in pixels.

    Returns:
        tuple: Tuple of (x0, y0, x1, y1) coordinates defining the bounding box of the crop region.
    """
    # Calculate the half size of the crop region
    crop_half_width = crop_size[1] // 2
    crop_half_height = crop_size[0] // 2

    # Calculate the x and y coordinates of the upper-left corner of the crop region
    x0 = max(coord[0] - crop_half_width, 0)
    y0 = max(coord[1] - crop_half_height, 0)

    # Calculate the x and y coordinates of the lower-right corner of the crop region
    x1 = min(coord[0] + crop_half_width, width)
    y1 = min(coord[1] + crop_half_height, height)


    # Return the crop coordinates as a tuple
    return (x0, y0, x1, y1)



def crop_image(image_path, coords, output_path):
    """
    Crop an image around a pair of coordinates and save the cropped image.

    Args:
        image_path (str): Path to the input image file.
        coords (tuple): Tuple of (x0, y0, x1, y1) coordinates defining the
            bounding box of the crop region.
        output_path (str): Path to save the cropped image file.
    """
    # Open the input image file
    with Image.open(image_path) as image:
        # Crop the image
        cropped_image = image.crop(coords)
        
        # Save the cropped image
        cropped_image.save(output_path)

def create_cropped_image(xy, img_path):

    data_dir = img_path.split('/')[:-1]
    img_id = img_path.split('.')[0].split('/')[-1].split('_')[-1].strip('0')
    output_path = f"{output_dir}/{img_id}_cropped.jpg"
    crop_image(img_path, get_crop_coords(xy, 1920, 1080,(128,320)), output_path)



def visualize_output(model, dataset, n_samples):
    model.eval()
    with torch.no_grad():
        losses = 0
        for i in range(n_samples):
            X, y, path = dataset.__getitem__(i)

            X,y = X.to(device), y.to(device)
            print(f"Target Coordinates: {y.detach().cpu().numpy()}")
            output = model(X.unsqueeze(0))
            # un-normalize output
            # un_normalize target
            # pass it to a function, which takes output coordinates and img_path
            # and save the cropped image in a folder
            output = un_normalize(output.detach().cpu().squeeze(0).numpy())
            create_cropped_image(output, path)



def test(model, optimizer, loss_function, dataset):

    # better have batch size equal to 16
    model.eval()
    with torch.no_grad():
        losses = 0
        data_loader = tqdm(dataset)
        for X, y, paths in data_loader:

            imgs_paths = list(paths)
            X,y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_function(output, y)
            data_loader.set_description(f'loss: :{loss.item()}')
            losses+=loss.item()
        # average loss
        test_loss = losses/len(data_loader)

    print(f"Average Loss of {len(data_loader)} number of samples is: {test_loss}")





model = TTNet(dropout_p=dropout,in_ch=3,num_frames_sequence=1,resize=(img_y, img_x),device=device)
model.load_state_dict(torch.load('model_checkpoints_curve/model_19.pth',map_location=torch.device('cpu')))
loss_function = F.mse_loss

transform = transforms.Compose([transforms.Resize([img_y,img_x],antialias=True),transforms.Normalize(mean=[0.485,0.456, 0.406], std=[0.229, 0.224, 0.225])])

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


dataset = TTDataset(
    f"{data_path}/test", transform=transform, size=(img_x, img_y), n_frames=0
)

# train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
test_loader = torch.utils.data.DataLoader(dataset,num_workers=2,batch_size=batch_size, collate_fn= collate_fn,shuffle=True,prefetch_factor=5)


# test(model, optimizer, loss_function, test_loader)

visualize_output(model,dataset,20)







