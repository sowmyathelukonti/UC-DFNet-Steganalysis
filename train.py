import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Import local modules
from models.ucdfnet import UCDFNet
from dataset.stego_dataset import StegoDataset, get_augmentations, generate_synthetic_dataset
from utils import plot_training_curves, plot_confusion_matrix

def train_model(args):
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Dataset Preparation
    data_dir = args.data_dir
    cover_dir = os.path.join(data_dir, "cover")
    stego_dir = os.path.join(data_dir, "stego")
    
    # Check if dataset exists, otherwise generate it
    if not (os.path.exists(cover_dir) and os.path.exists(stego_dir)):
        print("Dataset not found. Generating synthetic dataset...")
        generate_synthetic_dataset(data_dir, num_samples=args.num_samples)
        
    # Collect file paths and labels
    cover_files = [os.path.join(cover_dir, f) for f in os.listdir(cover_dir) if f.endswith('.png')]
    stego_files = [os.path.join(stego_dir, f) for f in os.listdir(stego_dir) if f.endswith('.png')]
    
    file_paths = cover_files + stego_files
    labels = [0] * len(cover_files) + [1] * len(stego_files)
    
    if len(file_paths) == 0:
        raise ValueError(f"No PNG images found in {cover_dir} or {stego_dir}!")
        
    # Split into train/validation sets (80% train, 20% validation)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Total training samples: {len(train_paths)}")
    print(f"Total validation samples: {len(val_paths)}")
    
    # Create datasets
    train_dataset = StegoDataset(train_paths, train_labels, transform=get_augmentations("train"))
    val_dataset = StegoDataset(val_paths, val_labels, transform=get_augmentations("val"))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 2. Model, Optimizer, Loss, Scheduler Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = UCDFNet(num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 3. Training Loop with Early Stopping
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels_batch.size(0)
            correct_train += (predicted == labels_batch).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels_batch.size(0)
                correct_val += (predicted == labels_batch).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_trues.extend(labels_batch.cpu().numpy())
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        # Calculate Validation Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_trues, val_preds, average='binary'
        )
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] -> "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | "
              f"F1 Score: {f1*100:.2f}%")
              
        # Learning Rate Scheduler step
        scheduler.step(epoch_val_loss)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), args.model_path)
            print(f"  --> Saved new best model checkpoint to '{args.model_path}'")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Early Stopping check
        if early_stop_counter >= args.patience:
            print(f"Early stopping triggered! Training stopped at epoch {epoch+1}.")
            break
            
    print("\nTraining completed.")
    
    # 4. Save Training Curves Plot
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir=".")
    print("Training curves saved as 'training_curves.png'")
    
    # 5. Evaluate Best Model & Create Confusion Matrix
    print(f"\nLoading best model '{args.model_path}' for final validation evaluation...")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    final_preds = []
    final_trues = []
    
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            final_preds.extend(predicted.cpu().numpy())
            final_trues.extend(labels_batch.numpy())
            
    # Calculate and save confusion matrix
    cm = confusion_matrix(final_trues, final_preds)
    plot_confusion_matrix(cm, classes=["Cover", "Stego"], save_path="confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Final Metrics Summary
    acc = accuracy_score(final_trues, final_preds)
    p, r, f, _ = precision_recall_fscore_support(final_trues, final_preds, average='binary')
    
    print("\n================== FINAL METRICS ==================")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {p*100:.2f}%")
    print(f"Recall:    {r*100:.2f}%")
    print(f"F1 Score:  {f*100:.2f}%")
    print("===================================================")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UC-DFNet Training Script for Color Image Steganalysis")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory where data is stored")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples per class to generate if dataset is missing")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--model-path", type=str, default="best_model.pth", help="Filepath to save the best model checkpoint")
    
    args = parser.parse_args()
    train_model(args)
