import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from features.built_features import BallDataset
from models.model import CombinedModel
import numpy as np
from icecream import ic as mprint


def get_crop_coords(x_center, y_center, w_resize=320, h_resize=128, w_original=1920, h_original=1080):
    x_min = max(0, x_center - int(w_resize / 2))
    y_min = max(0, y_center - int(h_resize / 2))

    x_max = min(w_original, x_min + w_resize)
    y_max = min(h_original, y_min + h_resize)

    return x_min, x_max, y_min, y_max

def denormalized(x, y):
    return int(x*1920), int(y*1080)

def within_bound(x_min, x_max, y_min, y_max, org_x, org_y):
    return (x_min< org_x <x_max and y_min < org_y < y_max)

def preprocess_global_out(batch, output_ball_coordinates):
    # crop coordinates
    for batch_idx, pred_coords in zip(np.arange(batch['image'].shape[0]), output_ball_coordinates):
        image, x, y  = batch['image'][batch_idx], batch["x"][batch_idx], batch["y"][batch_idx]
        # resize image
        image_resize = torch.nn.functional.interpolate(image.unsqueeze(0), 
                                                       size=(1080, 1920), 
                                                       mode='bilinear', 
                                                       align_corners=False
                                                       )
        pred_x, pred_y = denormalized(pred_coords[0], pred_coords[1])
        x_min, x_max, y_min, y_max = get_crop_coords(pred_x, pred_y)
        org_x, org_y = denormalized(x, y)
        is_detected = within_bound(x_min, x_max, y_min, y_max, org_x, org_y)
        if is_detected and ((x_max-x_min)==320 and (y_max-y_min)==128):
            batch['image'][batch_idx] = image_resize[:,:, y_min:y_max, x_min:x_max]
        else:
            # cropping image center
            batch['image'][batch_idx] = image_resize[:, :, int((1080/2)-(128/2)):int((1080/2)-(128/2)) + 128, int((1920/2)-(320/2)):int((1920/2)-(320/2)) + 320]
            batch['x'][batch_idx] = -1
            batch['y'][batch_idx] = -1

    return batch


# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your dataset and dataloader for inference
transform = transforms.Compose([
    transforms.ToTensor(),
])

csv_file_path_inference = '../input/test.csv'
inference_dataset = BallDataset(csv_file=csv_file_path_inference, transform=transform)
inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

# Instantiate your model (make sure to adjust the model architecture if needed)
in_ch = 3
dropout_p = 0.5
num_frames_sequence = 10
local = True  # Adjust according to your model's training setting

model = CombinedModel(in_ch, dropout_p, num_frames_sequence, local).to(device)

# Load the trained model (adjust the paths accordingly)
checkpoint_path = '../models/final_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

# Load the model state_dicts and optimizer state_dict
model.model1.load_state_dict(checkpoint['model_state_dict1'])
model.model2.load_state_dict(checkpoint['model_state_dict2'])

# Inference loop
model.eval()
with torch.no_grad():
    for batch in tqdm(inference_dataloader, desc="Inference"):
        images = batch['image'].to(device)

        # Forward pass for model1
        outputs_model1 = model.model1(images)
        batch = preprocess_global_out(batch, outputs_model1)

        # Input for model2
        nimages, nlabels_x, nlabels_y = batch['image'].to(device), batch['x'].to(device), batch['y'].to(device)

        # Forward pass for model2 using the modified inputs
        outputs_model2 = model.model2(nimages)
        # Assuming 'outputs_model2' contains predictions for x and y
        # You can use the predictions as needed for your specific application
        predicted_x, predicted_y = outputs_model2[0, 0], outputs_model2[0, 1]
        mprint(f"Predicted Coordinates: x={predicted_x*1920}, y={predicted_y*1080}")
        mprint(f"Actual Coordinates: x={nlabels_x.item()*1920}, y={nlabels_y.item()*1080}")
