import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

import zipfile

# Define zip path and extraction path
zip_path = "C:/Users/kamal/OneDrive/Desktop/Ayoub/Master/S2/Deep Learning/tp cnn kenet/amhcd-data-64.zip"
extract_dir = "C:/Users/kamal/OneDrive/Desktop/Ayoub/Master/S2/Deep Learning/tp cnn kenet/amhcd-data-64/amhcd-data-64/tifinagh-images"

# Unzip if not already done
if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print(f"Extracted dataset to: {extract_dir}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class LeNet5(nn.Module):
    """
    LeNet-5 CNN Architecture for Tifinagh Character Recognition
    Modified for 33 classes classification
    """
    def __init__(self, num_classes=33):
        super(LeNet5, self).__init__()
        
        # C1: Convolution layer - 6 filters of 5x5
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        
        # S2: Pooling layer - 2x2 average pooling with stride 2
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C3: Convolution layer - 16 filters of 5x5
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        # S4: Pooling layer - 2x2 average pooling with stride 2
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C5: Fully connected layer - 120 neurons
        self.c5 = nn.Linear(in_features=16*5*5, out_features=120)
        
        # F6: Fully connected layer - 84 neurons
        self.f6 = nn.Linear(in_features=120, out_features=84)
        
        # Output layer - 33 neurons for 33 classes
        self.output = nn.Linear(in_features=84, out_features=num_classes)
        
    def forward(self, x):
        # C1: Convolution + Activation
        x = torch.tanh(self.c1(x))  # 32x32x1 -> 28x28x6
        
        # S2: Pooling
        x = self.s2(x)  # 28x28x6 -> 14x14x6
        
        # C3: Convolution + Activation
        x = torch.tanh(self.c3(x))  # 14x14x6 -> 10x10x16
        
        # S4: Pooling
        x = self.s4(x)  # 10x10x16 -> 5x5x16
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # 5x5x16 -> 400
        
        # C5: Fully connected + Activation
        x = torch.tanh(self.c5(x))  # 400 -> 120
        
        # F6: Fully connected + Activation
        x = torch.tanh(self.f6(x))  # 120 -> 84
        
        # Output layer
        x = self.output(x)  # 84 -> 33
        
        return x

class TifinaghDataset(Dataset):
    """Custom Dataset for Tifinagh characters"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((
                            os.path.join(class_path, img_name),
                            self.class_to_idx[class_name]
                        ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_data_loaders(data_dir, batch_size=64, train_split=0.7, val_split=0.15):
    """Create train, validation, and test data loaders"""
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create dataset
    dataset = TifinaghDataset(data_dir, transform=transform)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset.classes

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=50):
    """Training function with validation"""
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct_train/total_train:.2f}%'
            })
        
        # Calculate average training metrics
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct_val / total_val
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print('-' * 60)
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def test_model(model, test_loader, criterion, class_names):
    """Test the model and return predictions"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_accuracy = 100. * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    return all_predictions, all_targets, test_accuracy

def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, data_loader, device, num_images=4):
    """Visualize feature maps from convolutional layers"""
    model.eval()
    
    # Get a batch of images
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images = images[:num_images].to(device)
    
    # Hook function to capture feature maps
    feature_maps = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook
    
    # Register hooks
    model.c1.register_forward_hook(hook_fn('C1'))
    model.c3.register_forward_hook(hook_fn('C3'))
    
    # Forward pass
    with torch.no_grad():
        _ = model(images)
    
    # Visualize feature maps
    for layer_name, fmaps in feature_maps.items():
        num_filters = fmaps.shape[1]
        fig, axes = plt.subplots(2, min(num_filters//2, 8), figsize=(16, 4))
        fig.suptitle(f'Feature Maps - {layer_name}')
        
        for i in range(min(num_filters, 16)):
            row = i // 8
            col = i % 8
            if num_filters > 8:
                ax = axes[row, col]
            else:
                ax = axes[col] if num_filters > 1 else axes
            
            fmap = fmaps[0, i].cpu().numpy()
            ax.imshow(fmap, cmap='viridis')
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def create_optimizer(model, optimizer_name='adam', lr=0.001):
    """Create optimizer"""
    if optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# Main execution
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "C:/Users/kamal/OneDrive/Desktop/Ayoub/Master/S2/Deep Learning/tp cnn kenet/amhcd-data-64/amhcd-data-64/tifinagh-images"
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    OPTIMIZER = 'adam'  # Options: 'sgd', 'adam', 'rmsprop'
    
    print("Setting up data loaders...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        DATA_DIR, BATCH_SIZE
    )
    
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Initialize model
    model = LeNet5(num_classes=len(class_names)).to(device)
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and loss function
    optimizer = create_optimizer(model, OPTIMIZER, LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nUsing optimizer: {OPTIMIZER}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    # Train the model
    print("\nStarting training...")
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS
    )
    
    # Test the model
    print("\nTesting the model...")
    predictions, targets, test_acc = test_model(model, test_loader, criterion, class_names)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Plot training curves
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    
    # Plot confusion matrix
    plot_confusion_matrix(targets, predictions, class_names)
    
    # Visualize feature maps
    visualize_feature_maps(model, test_loader, device)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(targets, predictions, target_names=class_names))
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': class_names,
        'test_accuracy': test_acc
    }, 'lenet5_tifinagh_model.pth')
    
    print(f"\nModel saved as 'lenet5_tifinagh_model.pth'")
    print(f"Final test accuracy: {test_acc:.2f}%")