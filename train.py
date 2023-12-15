import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
from features.built_features import BallDataset
from models.model import CombinedModel

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your dataset and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Adjust the path to your CSV file
csv_file_path = 'input/train.csv'
dataset = BallDataset(csv_file=csv_file_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate your model
in_ch = 3
dropout_p = 0.5
num_frames_sequence = 10
model = CombinedModel(in_ch, dropout_p, num_frames_sequence).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels_x, labels_y = batch['image'].to(device), batch['x'].to(device), batch['y'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Assuming 'outputs' contains predictions for x and y
        loss = criterion(outputs[:, 0], labels_x) + criterion(outputs[:, 1], labels_y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss at the end of each epoch
    average_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
