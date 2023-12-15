import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
from features.built_features import BallDataset
from models.model import CombinedModel
from icecream import ic as mprint
from data.create_dataset import create_dataset
import sys

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


if __name__ == "__main__":
    step = int(sys.argv[1])
    # step = 2
    # ... (your dataset creation code remains unchanged)
    # create dataset
    dataset_path = "/media/muneeb/New Volume1/projects/umair_jr_fyp_yolo/openttgames_dataset"
    # create train dataset
    create_dataset(os.path.join(dataset_path, "train"), os.path.join("/home/muneeb/Desktop/umair/fyp/ball-detectionv1/input", "train.csv"))
    # create test dataset
    create_dataset(os.path.join(dataset_path, "test"), os.path.join("/home/muneeb/Desktop/umair/fyp/ball-detectionv1/input", "test.csv"))
    
    # {x:0, y:0} means we have no ball coordinates for this

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define your dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    csv_file_path = '/home/muneeb/Desktop/umair/fyp/ball-detectionv1/input/train.csv'
    dataset = BallDataset(csv_file=csv_file_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Instantiate your model
    in_ch = 3
    dropout_p = 0.5
    num_frames_sequence = 10
    reload = True 
    local = True if step>1 else False

    model = CombinedModel(in_ch, dropout_p, num_frames_sequence, local).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if reload:
        saved_checkpoint_path = '../models/global_model.pth'
        if os.path.exists(saved_checkpoint_path):
            checkpoint = torch.load(saved_checkpoint_path)
            model.model1.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # start_epoch = checkpoint['epoch']

    f = open("/home/muneeb/Desktop/umair/fyp/ball-detectionv1/models/losses_model2.txt", "a")

    if step==1:
        # Training loop - Step 1: Train model1
        num_epochs = 15
        for epoch in range(start_epoch, start_epoch + num_epochs):
            model.train()
            running_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}")
            for batch in pbar:
                images, labels_x, labels_y = batch['image'].to(device), batch['x'].to(device), batch['y'].to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs[:, 0], labels_x) + criterion(outputs[:, 1], labels_y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                running_loss += loss.item()
                f.write(f"{loss.item()}\n")
                f.flush()

            # Print average loss at the end of each epoch
            average_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")

        # Save the trained model1
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.model1.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
        }, "../models/model1.pth")

    elif step==2:
        saved_checkpoint_path = '../models/global_model.pth'
        if os.path.exists(saved_checkpoint_path):
            checkpoint = torch.load(saved_checkpoint_path)
            model.model1.load_state_dict(checkpoint['model_state_dict'])
            model.model1.eval()
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # start_epoch = checkpoint['epoch']
            saved_checkpoint_path = '../models/model2.pth'
            checkpoint = torch.load(saved_checkpoint_path)
            model.model2.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

        num_epochs = 8
        # Step 2: Freeze model1, train model2
        model.model1.eval()  # Freeze model1
        model.model2.train()  # Set model2 to training mode

        # Define a new optimizer for model2
        optimizer_model2 = optim.Adam(model.model2.parameters(), lr=0.001)

        # New dataloader for model2 using the output of model1
        dataloader_model2 = DataLoader(dataset, batch_size=32, shuffle=True)

        # Training loop for model2
        for epoch in range(start_epoch, start_epoch + num_epochs):
            model.train()
            running_loss = 0.0
            pbar = tqdm(dataloader_model2, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}")
            for batch in pbar:
                images, labels_x, labels_y = batch['image'].to(device), batch['x'].to(device), batch['y'].to(device)

                # Forward pass using model1
                outputs_model1 = model.model1(images)
                batch = preprocess_global_out(batch, outputs_model1)
                # input for model2
                nimages, nlabels_x, nlabels_y = batch['image'].to(device), batch['x'].to(device), batch['y'].to(device)

                # Forward pass for model2 using the modified inputs
                outputs_model2 = model.model2(nimages)

                # Assuming 'outputs_model2' contains predictions for x and y
                loss_model2 = criterion(outputs_model2[:, 0], nlabels_x) + criterion(outputs_model2[:, 1], nlabels_y)

                # Backward pass and optimization for model2
                optimizer_model2.zero_grad()
                loss_model2.backward()
                optimizer_model2.step()

                pbar.set_postfix(loss=f"{loss_model2.item():.4f}")
                running_loss += loss_model2.item()
                f.write(f"{loss_model2.item()}\n")
                f.flush()

            # Print average loss at the end of each epoch for model2
            average_loss_model2 = running_loss / len(dataloader_model2)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss (Model2): {average_loss_model2}")

        # Save the trained model2
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.model2.state_dict(),
            'optimizer_state_dict': optimizer_model2.state_dict(),
            'loss': average_loss_model2,
        }, "../models/model2.pth")

    elif step==3:
        saved_checkpoint_path = '../models/final_model.pth'
        if not os.path.exists(saved_checkpoint_path):
            checkpoint = torch.load(saved_checkpoint_path)
            model.model1.load_state_dict(checkpoint['model_state_dict1'])
            model.model2.load_state_dict(checkpoint['model_state_dict2'])
            start_epoch = checkpoint['epoch']
        else:
            checkpoint = torch.load(saved_checkpoint_path.replace("/final_model", "/global_model"))
            model.model1.load_state_dict(checkpoint['model_state_dict'])
            checkpoint = torch.load(saved_checkpoint_path.replace("/final_model", "/model2"))

            model.model2.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
        num_epochs = 10
        # Step 3: Unfreeze both models and train them together
        model.model1.train()
        model.model2.train()

        # Create a new optimizer for both models
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop for both models
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}")
            for batch in pbar:
                images, labels_x, labels_y = batch['image'].to(device), batch['x'].to(device), batch['y'].to(device)

                outputs_model1 = model.model1(images)
                batch = preprocess_global_out(batch, outputs_model1)
                # input for model2
                nimages, nlabels_x, nlabels_y = batch['image'].to(device), batch['x'].to(device), batch['y'].to(device)

                # Forward pass for model2 using the modified inputs
                outputs_model2 = model.model2(nimages)

                # Assuming 'outputs_model2' contains predictions for x and y
                loss_model1 = criterion(outputs_model1[:, 0], labels_x) + criterion(outputs_model1[:, 1], labels_y)

                loss_model2 = criterion(outputs_model2[:, 0], nlabels_x) + criterion(outputs_model2[:, 1], nlabels_y)

                # Backward pass and optimization for model2
                optimizer.zero_grad()

                # Total loss is the sum of losses for both models
                total_loss = loss_model1 + loss_model2

                # Backward pass and optimization for both models
                total_loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=f"{total_loss.item():.4f}")
                running_loss += total_loss.item()
                f.write(f"{total_loss.item()}\n")
                f.flush()

            # Print average loss at the end of each epoch for both models
            average_loss_both = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss (Model1 + Model2): {average_loss_both}")

        # Save the final trained model (both models)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict1': model.model1.state_dict(),
            'model_state_dict2': model.model2.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss_both,
        }, "../models/final_model.pth")

    f.close()
