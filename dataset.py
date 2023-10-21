import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, csv):
        self.csv = csv
        self.all_image_names = self.csv[:]['Id']
        self.all_labels = np.array(self.csv.drop(['Id', 'Genre'], axis=1))

        # Define the transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((400, 400)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
        ])
        

    def __len__(self):
        return len(self.all_image_names)
    
    def __getitem__(self, index):
        image = cv2.imread(f"C:/Users/Karti/Desktop/drive/Multi_Label_dataset/Images/{self.all_image_names[index]}.jpg")
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.all_labels[index]
        
        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }
