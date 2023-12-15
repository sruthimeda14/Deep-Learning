import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from icecream import ic as mprint
import os
from models.model import CombinedModel  # Import your CombinedModel class
from features.built_features import BallDataset  # Import your BallDataset class

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your test dataset and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Adjust the path to your test CSV file
test_csv_file_path = '../input/test.csv'
test_dataset = BallDataset(csv_file=test_csv_file_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate your model
in_ch = 3
dropout_p = 0.5
num_frames_sequence = 10
model = CombinedModel(in_ch, dropout_p, num_frames_sequence).to(device)

# Load the trained model state
saved_model_path = '../models/global_model.pth'
checkpoint = torch.load(saved_model_path)

# Load the model and optimizer states
model.model1.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode

# Inference loop for the test set
predictions = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Inference on Test Set"):
        images, _, _ = batch['image'].to(device), batch['x'].to(device), batch['y'].to(device)

        # Forward pass
        outputs = model(images)

        # Assuming 'outputs' contains predictions for x and y
        predictions.extend(outputs.cpu().numpy())


# Visualize the predictions on images
for i in range(len(test_dataset)):
    image, true_x, true_y = test_dataset[i]['image'], test_dataset[i]['x'], test_dataset[i]['y']
    pred_x, pred_y = predictions[i]

    # Convert image to numpy array
    image_np = transforms.ToPILImage()(image).convert("RGB")
    mprint(abs(true_x*1920-pred_x*1920))
    #mprint(pred_x*1920)

    # mprint(true_y*1080)
    # mprint(pred_y*1080)
    # Plot true coordinates (in red) and predicted coordinates (in blue) on the image
    # plt.imshow(image_np)
    # plt.scatter(true_x, true_y, color='red', marker='o', label='True Coordinates')
    # plt.scatter(pred_x*1920, pred_y*1080, color='blue', marker='x', label='Predicted Coordinates')
    # plt.legend()
    # plt.title(f"Image {i+1}")
    # plt.show()
