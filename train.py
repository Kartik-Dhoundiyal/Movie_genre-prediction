import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from engine import train, validate
from dataset import ImageDataset
from torch.utils.data import DataLoader
from models import model_with_attention  # Import your custom model function

# Initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model with attention
pretrained = True
requires_grad = False  # Set to True if you want to fine-tune the model
num_attention_heads = 8  # Adjust this number based on your needs
model = model_with_attention(pretrained, requires_grad, num_attention_heads).to(device)

# Learning parameters
lr = 0.0001
epochs = 20
batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# Read the training csv file and validation csv file
train_csv = pd.read_csv('C:/Users/Karti/Desktop/drive/Multi_Label_dataset/train.csv')
val_csv = pd.read_csv('C:/Users/Karti/Desktop/drive/Multi_Label_dataset/val.csv')  # Replace with the path to your validation CSV

# Initialize datasets
train_data = ImageDataset(train_csv)
val_data = ImageDataset(val_csv)  # Use the validation dataset

# Initialize data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Start the training and validation
train_loss = []
val_loss = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(model, train_loader, optimizer, criterion, train_data, device)
    val_epoch_loss = validate(model, val_loader, criterion, val_data, device)
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Validation Loss: {val_epoch_loss:.4f}')

# Save the trained model to disk
torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
}, 'C:/Users/Karti/Desktop/drive/outputs/model.pth')

# Plot and save the train and validation line graphs for losses
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='Train Loss')
plt.plot(val_loss, color='red', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('C:/Users/Karti/Desktop/drive/outputs/loss.png')
plt.show()
