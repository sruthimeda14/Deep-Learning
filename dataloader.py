from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from PIL import Image
import json
import os
from matplotlib import pyplot as plt
import cv2
from copy import deepcopy
import torchvision.transforms as transforms
import zipfile
from matplotlib import pyplot as plt
import cv2
from torchvision.io import read_image
# https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


# /media/umair/60db4ff9-b102-4dab-8258-101c357958ab/dataset/openttgames_kaggle/archive/dataset/train




def preprocess_annotations(root_dir):

    # {"file_name", "event"}

    markups_path = [os.path.join(root_dir, dir,"ball_markup.json")  for dir in os.listdir(root_dir)]
    images_dir_path = [os.path.join(root_dir, dir,"images")  for dir in os.listdir(root_dir)]
    # open all the files
    all_markups = []
    for path, imgs in zip(markups_path,images_dir_path):
        imgs_list = os.listdir(imgs)
        fp = open(path, "r")
        data_dir = path.split('/')[:-1]
        data = json.load(fp)
        all_markups.extend(
            [os.path.join('/',*data_dir, "images/img_"+key.zfill(6)+".jpg"), [value['x'],value['y']]]
            for key, value in data.items() if "img_"+key.zfill(6)+".jpg" in imgs_list
        )
            # create file name

    return all_markups

# dir structure
# dataset
#       |--- train
#               |-----game_1 ==> images, masks, _.json, _.json
#               |-----game_2 ==> images, masks, _.json, _.json
#       |--- test
#               |-----test_1 ==> images, masks, _.json, _.json


class TTDataset(Dataset):
    def __init__(self, dir, transform, size=(128,320), n_frames=4):
        super(TTDataset).__init__()


        self.transform = transform
        self.dir_path = dir
        self.markups = preprocess_annotations(dir)
        # self.length = length
        self.n_frames = n_frames # if we want to take 9 frames, then it is 4
        self.img_size_new = size
        self.img_size = (1920,1080)
    def __len__(self):

        t = len(self.markups)*0.5
        return int(t)
    

    def __normalize__(self, pos_xy):
        if pos_xy[0] ==-1 or pos_xy[1]==-1:
          return np.array([-1,-1]) 
        # x = self.img_size[0]
        # y = self.img_size[1]
        # x_ratio = x / self.img_size_new[0]
        # y_ratio = y / self.img_size_new[1]
        # ball_position_xy = np.array([pos_xy[0] / x_ratio,pos_xy[1] / y_ratio])


        return [pos_xy[0]/self.img_size[0], pos_xy[1]/self.img_size[1]]
    

    def __getitem__(self, idx):


        data = self.markups[idx] # it is a list, data[0] = path and data[1] = [x,y]

        try:
            img = self.transform(read_image(data[0]).float()/255)
            t = self.__normalize__(data[1])#.reshape((1,2))
            ball_xy_pos = torch.tensor(t).float()
            return (img, ball_xy_pos, data[0])
        except Exception:
            return None

        
        


if __name__ == '__main__':
    path = "/content/content/dataset/train"
    img_x, img_y = 320,128  # resize video 2d frame size

    transform = transforms.Compose([transforms.Resize([img_y, img_x],antialias=True)])#,transforms.Normalize(mean=(0.5,), std=(0.5,))
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    d = TTDataset(path,transform,(img_x,img_y),0)
    for i in range(10):
      img, c, p = d.__getitem__(i) 
      print(c)
      print(p)









