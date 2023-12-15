from data.create_dataset import create_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
from features.built_features import BallDataset
from models.model import CombinedModel
from icecream import ic as mprint


if __name__ == "__main__":
    # create dataset
    dataset_path = "/media/muneeb/New Volume1/projects/umair_jr_fyp_yolo/openttgames_dataset"
    # create train dataset
    create_dataset(os.path.join(dataset_path, "train"), os.path.join("../input/", "train.csv"))
    # create test dataset
    create_dataset(os.path.join(dataset_path, "test"), os.path.join("../input/", "test.csv"))
    
    # {x:0, y:0} means we have no ball coordinates for this

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define your dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Adjust the path to your CSV file
    csv_file_path = '../input/train.csv'
    dataset = BallDataset(csv_file=csv_file_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Instantiate your model
    in_ch = 3
    dropout_p = 0.5
    num_frames_sequence = 10
    reload = True 
    model = CombinedModel(in_ch, dropout_p, num_frames_sequence).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if reload:
        saved_checkpoint_path = '../models/global_model.pth'
        if os.path.exists(saved_checkpoint_path):
            checkpoint = torch.load(saved_checkpoint_path)
            model.model1.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {start_epoch + 1}")

    f = open("../models/losses.txt", "a")
    # Training loop
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
            # mprint(outputs[:, 0], labels_x)
            # mprint(outputs[:, 1], labels_y)
            # Assuming 'outputs' contains predictions for x and y
            loss = criterion(outputs[:, 0], labels_x) + criterion(outputs[:, 1], labels_y)
            # mprint(outputs[:,0]*1920)
            # mprint(labels_x*1920)
            # mprint(outputs[:,1]*1080)
            # mprint(labels_y*1080)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            running_loss += loss.item()
            # losses.append(loss.item())
            f.write(f"{loss.item()}\n")
            f.flush()

        # Print average loss at the end of each epoch
        average_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")

    # Save the trained model
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.model1.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': average_loss,
    }, "../models/global_model.pth")

