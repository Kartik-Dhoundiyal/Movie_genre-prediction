import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from engine import validate
from dataset import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import models

matplotlib.style.use('ggplot')

# Initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = models.model(pretrained=True, requires_grad=False).to(device)

# Load the trained model checkpoint
checkpoint = torch.load('C:/Users/Karti/Desktop/drive/outputs/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Read the train csv file, as it contains the test split
train_csv = pd.read_csv('C:/Users/Karti/Desktop/drive/Multi_Label_dataset/train.csv')

# Initialize the test dataset using the 'test' split
test_data = ImageDataset(train_csv, split='test')

# Initialize the data loader for the test dataset
batch_size = 32  # You can adjust this batch size as needed
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define a function to calculate accuracy
def calculate_accuracy(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in dataloader:
            images, targets = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            outputs = (torch.sigmoid(outputs) > 0.5).float()  # Threshold outputs
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# Calculate accuracy on the test set
test_accuracy = calculate_accuracy(model, test_loader, device)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plot and save the test accuracy graph
plt.figure(figsize=(10, 7))
plt.plot([test_accuracy], color='purple', marker='o', label='Test Accuracy')
plt.xlabel('Test')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('C:/Users/Karti/Desktop/drive/outputs/test_accuracy.png')
plt.show()
