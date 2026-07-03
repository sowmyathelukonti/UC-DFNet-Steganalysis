import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import string
import sys

# Ensure parent directory is in path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import embed_lsb, embed_lsb_matching, embed_random_path, embed_dct

class StegoDataset(Dataset):
    """
    PyTorch Dataset for Color Image Steganalysis.
    Loads cover and stego images, and applies data augmentation.
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load image (OpenCV loads BGR, we convert to RGB for standard PyTorch processing)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for torchvision transforms
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


def get_augmentations(split="train"):
    """
    Returns data augmentation and preprocessing transforms.
    Includes rotation, flipping, brightness adjustment, random cropping, and normalization.
    """
    if split == "train":
        return transforms.Compose([
            # Data Augmentation
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.1),
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
            # Preprocessing
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # 'val' or 'test'
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def generate_random_string(length=50):
    """Generate a random alphanumeric string."""
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))


def generate_synthetic_dataset(data_dir="data", num_samples=100, img_size=(256, 256)):
    """
    Generates a synthetic dataset of Cover (Class 0) and Stego (Class 1) images.
    Creates beautiful complex patterns with high-frequency noise and embeds messages using LSB.
    """
    cover_dir = os.path.join(data_dir, "cover")
    stego_dir = os.path.join(data_dir, "stego")
    
    os.makedirs(cover_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)
    
    print(f"Generating synthetic dataset in '{data_dir}' with {num_samples} samples per class...")
    
    for i in range(num_samples):
        # 1. Generate a complex textured cover image
        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        
        # Draw background gradient
        color1 = [random.randint(0, 255) for _ in range(3)]
        color2 = [random.randint(0, 255) for _ in range(3)]
        for y in range(img_size[0]):
            alpha = y / img_size[0]
            color = [int(c1 * (1 - alpha) + c2 * alpha) for c1, c2 in zip(color1, color2)]
            img[y, :] = color
            
        # Draw random shapes (circles, lines, rectangles)
        num_shapes = random.randint(3, 8)
        for _ in range(num_shapes):
            shape_type = random.choice(["circle", "line", "rectangle"])
            shape_color = [random.randint(0, 255) for _ in range(3)]
            thickness = random.choice([-1, 1, 2, 3]) # -1 means filled for circle/rect
            
            if shape_type == "circle":
                center = (random.randint(0, img_size[1]), random.randint(0, img_size[0]))
                radius = random.randint(10, 80)
                cv2.circle(img, center, radius, shape_color, thickness)
            elif shape_type == "line":
                pt1 = (random.randint(0, img_size[1]), random.randint(0, img_size[0]))
                pt2 = (random.randint(0, img_size[1]), random.randint(0, img_size[0]))
                cv2.line(img, pt1, pt2, shape_color, max(1, thickness))
            elif shape_type == "rectangle":
                pt1 = (random.randint(0, img_size[1]), random.randint(0, img_size[0]))
                pt2 = (random.randint(0, img_size[1]), random.randint(0, img_size[0]))
                cv2.rectangle(img, pt1, pt2, shape_color, thickness)
                
        # Add high-frequency Gaussian noise to simulate sensor noise (makes steganalysis more realistic)
        noise = np.random.normal(0, random.uniform(3, 8), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save Cover Image
        cover_path = os.path.join(cover_dir, f"cover_{i:04d}.png")
        cv2.imwrite(cover_path, img)
        
        # 2. Embed secret message to create Stego Image using ONLY Random Path LSB
        message = f"UC-DFNet_Stego_Payload_ID_{i:04d}_{generate_random_string(80)}"
        key = random.randint(1, 1000)
        channels_opt = random.choice([[0, 1, 2], [0], [1], [2]])  # All, Red, Green, or Blue
        stego_img = embed_random_path(img, message, key=key, channels=channels_opt)
        
        # Save Stego Image
        stego_path = os.path.join(stego_dir, f"stego_{i:04d}.png")
        cv2.imwrite(stego_path, stego_img)
        
    print("Dataset generation complete!")
