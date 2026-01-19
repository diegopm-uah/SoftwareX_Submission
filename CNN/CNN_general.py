import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import  matplotlib.pyplot   as plt
import math
import logging
# from pytorch3d.loss import chamfer_distance
import csv
import time
import argparse
from datetime import datetime
from torcheval.metrics.functional import multiclass_confusion_matrix
import json
from pathlib import Path
import pandas as pd
# from torchsummary import summary

# Getting day and time
actual_time = datetime.now()
torch.cuda.empty_cache()

# Getting the user and setting the path and gpu correctly
userPath = os.getcwd().split('/')[2]
cuda_own = "cuda:0"
if userPath == "newfasant2":
    userPath = userPath + "/N101"
    cuda_own = "cuda:1"

logs_folder_path=f'/home/{userPath}/N101-IA/CNN/Logs'
os.makedirs(logs_folder_path, exist_ok=True)
logging.basicConfig(
    filename=f'/home/{userPath}/N101-IA/CNN/Logs/Day_{actual_time.day}_{actual_time.month}_{actual_time.year}_Time_{actual_time.hour:02d}_{actual_time.minute:02d}_NN.log',  # Name of the log file
    level=logging.INFO,  # Logs level (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

parser = argparse.ArgumentParser(description='Procesar argumentos para enviarlos a los distintos programas necesarios.')
# parser.add_argument('-i', '--input_path', type=str, help='Ruta de la carpeta en la que se encuentra el dataset (terminada en una carpeta formato Classification_660_0_64_f_64_d).')
parser.add_argument('-u', '--use_case', type=str, default='Classification', choices=['Classification', 'Regression'], help='Tipo de uso que deseado  a la red')
# Below, npy refers to the field matrix, fft refers to the FFT of the field matrix and ISAR refers to the ISAR images generated from the field matrix.
# parser.add_argument('-d', '--data_type', type=str, default='ISAR', choices=['npy','fft','ISAR','fft_perfil','perfil_png','field','field_amp_npy','field_ph_npy','field_amp_png','field_ph_png'], help='Data type to be used to train the network.')
parser.add_argument('-e', '--num_epochs', type=int,  help='Number of epochs.')

args = parser.parse_args()

print(f'\nLos argumentos son {vars(args)}\n')
logging.info(f'Los argumentos son {vars(args)}\n')

device = (
    cuda_own
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
         )

torch.cuda.empty_cache()

seed = 110 # Setting the seed to be used in every part of the code

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


print(f"\nUsing {device} device and seed {seed}.\n")
# logging.info(f"\nUsing {device} device, {args.data_type} data type, {args.use_case} use case and seed {seed}.\n")


# class CustomDataset(Dataset):
#     """Custom dataset for loading data files and their corresponding labels.

#     Initializes the CustomDataset instance.

#     Args:
#         npy_dir (str): Directory containing the data files.
#         labels_file (str): File path to the file containing labels.
#         transform_ISAR (callable, optional): Optional transform to be applied on a sample for ISAR data type.
#         transform_npy (callable, optional): Optional transform to be applied on a sample for npy data type.
#     """

#     def __init__(self, dir, transform_ISAR=None, transform_npy=None):

#         self.dir = dir

#         if args.data_type == 'npy':
#             # List all .npy files in the directory.
#             self.file_names = [f for f in os.listdir(dir) if f.endswith('.npy') and f.startswith('sample_') and not f.endswith('fft.npy') and not f.endswith('perfil.npy') and not f.endswith('field.npy') and not f.endswith('field_amp.npy') and not f.endswith('field_ph.npy')]
#             self.transform = transform_npy
#         elif args.data_type == 'fft':
#             # List all fft files in the directory.
#             self.file_names = [f for f in os.listdir(dir) if f.endswith('fft.npy')]
#             self.transform = transform_npy
#         elif args.data_type == 'fft_perfil':
#             # List all fft (in profiles) files in the directory.
#             self.file_names = [f for f in os.listdir(dir) if f.endswith('perfil.npy')]
#             self.transform = transform_npy
#         elif args.data_type == 'field':
#             # List all field files in the directory.
#             self.file_names = [f for f in os.listdir(dir) if f.endswith('field.npy')]
#             self.transform = transform_npy
#         elif args.data_type == 'field_amp_npy':
#             # List all files containing the amplitude of the field in the directory.
#             self.file_names = [f for f in os.listdir(dir) if f.endswith('field_amp.npy')]
#             self.transform = transform_npy
#         elif args.data_type == 'field_ph_npy':
#             # List all files containing the complex phase of the field in the directory.
#             self.file_names = [f for f in os.listdir(dir) if f.endswith('field_ph.npy')]
#             self.transform = transform_npy
#         elif args.data_type == 'perfil_png':
#             # List all the images plotting the fft (in profiles) in the directory.
#             self.file_names = [f for f in os.listdir(dir) if f.endswith('perfil.png')]
#             self.transform = transform_ISAR
#         elif args.data_type == 'field_amp_png':
#             # List all the images plotting the amplitude of the field in the directory.
#             self.file_names = [f for f in os.listdir(dir) if f.endswith('amp.png')]
#             self.transform = transform_ISAR
#         elif args.data_type == 'field_ph_png':
#             # List all the images plotting the complex phase of the field in the directory.
#             self.file_names = [f for f in os.listdir(dir) if f.endswith('ph.png')]
#             self.transform = transform_ISAR
#         elif args.data_type == 'ISAR':
#             # List all the ISAR images in the directory.
#             self.file_names = [f for f in os.listdir(dir) if f.endswith('.png') and not f.endswith('perfil.png') and not f.endswith('amp.png') and not f.endswith('ph.png')]
#             self.transform = transform_ISAR
            
#         if args.use_case == "Classification":
#             # The network must say which of the geometries corresponds to a particular input
#             _, _, result = dir.split('/')[-1].partition('_')  # Divides in 3 parts: before, splitter and after
#             result = f"_{result}"
#             # Load labels vector from the labels file.
#             self.labels_vector = np.load(dir + '/labels_vector' + result + '.npy')
#         elif args.use_case == "Regression":
#             # The network must output the coordinates of the object of a particular input
#             self.coords = np.load(dir + "/coords.npy")[:, 1:]
#             self.dist_max = self.dist_max_calc()
            
#         # Sort files based on the sample id which is the number right after "sample_". For example, "sample_5.npy" will be split into ['sample', '5', '.npy'].
#         self.file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            
#     def __len__(self):
#         """Returns the total number of samples in the dataset.

#         Returns:
#             int: Number of samples.
#         """
#         return len(self.file_names)
    
#     def __getitem__(self, idx):

#         # Construct the full file path for the sample.
#         file_path = os.path.join(self.dir, self.file_names[idx])
        
#         if args.data_type == 'npy' or args.data_type == 'fft' or args.data_type == 'fft_perfil' or args.data_type == 'field' or args.data_type == 'field_amp_npy' or args.data_type == 'field_ph_npy':
#             # Load the numpy array with shape (2 x n x m).
#             data = np.load(file_path)
#             data = np.transpose(data, (2, 0, 1))
            
#             # Convert the numpy array to a PyTorch tensor.
#             data = torch.tensor(data, dtype=torch.float32).to(device)
#         elif args.data_type == 'ISAR' or args.data_type == 'perfil_png' or args.data_type == 'field_amp_png' or args.data_type == 'field_ph_png':
#             # Load the image and convert it to "L"
#             data = Image.open(file_path).convert("L")
            
#             # If a transform is set, apply it
#             if self.transform:
#                 data = self.transform(data).to(device)
#             else:
#                 data.to(device)

#         if args.use_case == "Classification":
#             # Load labels vector from the labels file.
#             target = torch.tensor(self.labels_vector[idx], dtype=torch.float32).to(device)
#         elif args.use_case == "Regression":
#             # Load coordinates from the coords matrix file.
#             target = torch.tensor(self.coords[idx,:], dtype=torch.float32).to(device)
        
#         return data, target
    
#     def dist_max_calc(self):
#         # Returns maximux size of the coordinates vector of the points
#         sample_coords = self.coords[1].reshape(int(self.coords.shape[1]/3),3)
#         dist_max = 0.0
#         for i in range(len(sample_coords)):
#             if sum(sample_coords[i] ** 2) > dist_max:
#                 dist_max = sum(sample_coords[i] ** 2)
#             dist_max = np.sqrt(dist_max)
#         return dist_max

class CustomDataset(Dataset):
    """
    Modern dataset that loads from a 'manifest.csv' generated by the dataset tool.
    Handles both ISAR images (.png) and Raw Complex Data (.npy) automatically.
    """
    def __init__(self, transform_ISAR=None, transform_npy=None):
        """
        Args:
            dataset_root (str): Path to the dataset folder (containing 'manifest.csv').
            transform_ISAR (callable): Transforms for images (e.g., Resize, ToTensor).
            transform_npy (callable): Transforms for raw tensors (e.g., Normalization).
        """
        dataset_root = Path("labeled_dataset")
        self.root = dataset_root
        self.manifest_path = os.path.join(dataset_root, 'manifest.csv')
        self.transform_ISAR = transform_ISAR
        self.transform_npy = transform_npy

        # 1. Load the Manifest
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}. Run generator first.")
        
        self.manifest = pd.read_csv(self.manifest_path, sep=';')
        
        print(f"Total Samples: {len(self.manifest)}, Geometries: {self.manifest['label_idx'].nunique()}")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):

        row = self.manifest.iloc[idx]
        file_path = row['file_path']
        label = int(row['label_idx'])
        
        data = None

        # ISAR Image (.png)
        if file_path.endswith('.png'):
            image = Image.open(file_path).convert('L')
            
            if self.transform_ISAR:
                data = self.transform_ISAR(image)
            else:
                # Default: Convert to tensor [1, H, W] and normalize 0-1
                data = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0

        # Raw Data (.npy) -> RCS, Field.
        elif file_path.endswith('.npy'):
            data_np = np.load(file_path)
            data_tensor = torch.from_numpy(data_np).float()
            
            # If shape is (H, W) -> Add Channel -> (1, H, W)
            # If shape is (2, H, W) -> Keep as is (Complex Real/Imag)
            if data_tensor.ndim == 2:
                data_tensor = data_tensor.unsqueeze(0)
            
            if self.transform_npy:
                data = self.transform_npy(data_tensor)
            else:
                data = data_tensor

        return data.to(device), torch.tensor(label, dtype=torch.long).to(device)

class CNNModule(nn.Module):
    """Convolutional module to extract features from a 2-channel input.

    This network processes an input tensor with shape (batch_size, 2, height, width)
    and returns a flattened feature vector.
    """

    def __init__(self):
        super(CNNModule, self).__init__()
        
        # Setting the neural networks layers
        # Convolution
        self.conv1 = nn.LazyConv2d(out_channels=96, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.LazyConv2d(256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.LazyConv2d(384, kernel_size=3, padding=1)
        self.conv4 = nn.LazyConv2d(384, kernel_size=3, padding=1)
        self.conv5 = nn.LazyConv2d(256, kernel_size=3, padding=1)
        # Pooling
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # ReLu and flattening
        self.ReLU = nn.ReLU()
        self.Flatten = nn.Flatten()

    def forward(self, x):
        # Applying the layers in the correct order
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.conv4(x)
        x = self.ReLU(x)
        x = self.conv5(x)
        x = self.ReLU(x)
        x = self.pool3(x)

        x = self.gap(x)

        x = self.Flatten(x)
        return x
    
# class GeometryPredictor(nn.Module):

    # def __init__(self, num_labels=None, coords_width=None):

    #     super(GeometryPredictor, self).__init__()
        
    #     self.cnn = CNNModule()

    #     # Fully connected layers
    #     self.fc1 = nn.LazyLinear(4096)  # Adjust size if needed.
    #     self.fc2 = nn.LazyLinear(4096)  #
        
    #     if args.use_case == "Classification":
    #         self.fc3 = nn.LazyLinear(num_labels)  # Number of outputs equal to num_labels.
    #     if args.use_case == "Regression":
    #         self.fc3 = nn.LazyLinear(coords_width)  # 3*N outputs, one for each coordinate (x, y, z) of the N vertices

    # def forward(self, x):

    #     # Process the input through the CNN to extract features.
    #     features = self.cnn(x)

    #     # Pass the features through fully connected layers.
    #     x = F.relu(self.fc1(features))
    #     x = F.relu(self.fc2(x))
    #     # Return the final results of the last layer of the network
    #     output = self.fc3(x)

    #     return output

class GeometryPredictor(nn.Module):

    def __init__(self, num_labels=None, coords_width=None):

        super(GeometryPredictor, self).__init__()
        
        self.cnn = CNNModule()
        
        # We can define the layers directly without 'Lazy'
        self.fc1 = nn.Linear(256, 512) # 256 in, 512 out
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, 512) # 512 in, 512 out
        self.relu2 = nn.ReLU()

        # Define fc3 based on the use case
        if args.use_case == "Classification":
            self.fc3 = nn.Linear(512, num_labels)  # 512 in, num_labels out
        if args.use_case == "Regression":
            self.fc3 = nn.Linear(512, coords_width) # 512 in, coords_width out

    def forward(self, x):

        # Process the input through the CNN to extract features.
        # This now outputs a tensor of shape (batch_size, 256)
        features = self.cnn(x)

        # Pass the features through fully connected layers.
        x = self.fc1(features)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)

        # Return the final results of the last layer of the network
        output = self.fc3(x)

        return output

class EarlyStopping:

    def __init__(self, patience=5, min_delta_percent=0.0, start_epoch=0):
        """
        Args:
            patience (int): How many epochs to wait before stopping if no improvement.
            min_delta_percent (float): Minimum *percentage* change in loss to be considered as improvement.
                                       E.g., 0.01 means a 1% decrease is required.
            start_epoch (int): Epoch number to start monitoring for early stopping.
        """
        self.patience = patience
        self.min_delta_percent = min_delta_percent
        self.start_epoch = start_epoch
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        """
        Call this method at the end of each epoch.

        Args:
            val_loss (float): Current validation loss.
            epoch (int): Current epoch number (0-based or 1-based depending on your convention).

        Returns:
            bool: True if training should stop early, False otherwise.
        """
        # Don't start early stopping until a certain start_epoch
        if epoch < self.start_epoch:
            return False

        required_improvement = self.best_loss * (1 - self.min_delta_percent)

        # Check for improvement
        if val_loss < required_improvement:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        # Trigger early stopping if patience exceeded    
        if self.counter >= self.patience:
            print(f"\nEarly stopping triggered after {self.patience} epochs.\n")
            logging.info(f"\nEarly stopping triggered after {self.patience} epochs.\n")

            return True

        return False

transform_I = transforms.Compose([
    transforms.Resize((128, 128)),   # Change this depending of image sizes
    transforms.ToTensor()            # Converting images to tensors
])

dataset = CustomDataset(transform_ISAR=transform_I)

if args.use_case == "Classification":
    # Initializing the model
    model = GeometryPredictor(num_labels=dataset.manifest['label_idx'].nunique()).to(device)

    # Setting the optimizer and the loss function
    optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
    criterion_CEL = nn.CrossEntropyLoss()
    
elif args.use_case == "Regression":
    # Intitializing the model
    model = GeometryPredictor(coords_width=dataset.coords.shape[1], num_labels=dataset.manifest['label_idx'].nunique()).to(device)

    # Setting the optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split the dataset into train, validation and test
train_size = int(0.7 * len(dataset))  # 70% for train
val_size = int(0.15 * len(dataset))  # 15% for validation
test_size = len(dataset) - train_size - val_size  # 15% for test

generator = torch.Generator().manual_seed(seed)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

# Creating dataloaders to iterate over datasets
train_batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_batch_size*32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_batch_size*32, shuffle=False)

# Early Stopping Instance
early_stopping = EarlyStopping(patience=8, min_delta_percent=0.005 ,start_epoch=int(0.2*args.num_epochs))

# Initializing empty vectors where losses will be saved
num_train_batches = len(train_loader)
num_val_batches = len(val_loader)
num_test_batches = len(test_loader)
avg_val_loss = np.empty(args.num_epochs, dtype=np.float32)
avg_train_loss = np.empty(args.num_epochs, dtype=np.float32)
avg_test_loss = np.empty(args.num_epochs, dtype=np.float32)

for epoch in range(args.num_epochs):
    start_time = time.time()
    running_train_loss = 0.0
    running_val_loss = 0.0
    
    model.train()
    if args.use_case == "Classification":
        for data, target_label in train_loader:

            optimizer.zero_grad()
            # Forward pass
            output_logits = model(data) # This output gives the logits, not the probabilities
            pred_probs= F.softmax(output_logits, dim=1) # This converts the logits to probabilities

            loss = criterion_CEL(output_logits, target_label.long())
            running_train_loss += loss.item()

            # Backward pass and optimizing
            loss.backward()
            optimizer.step()
    
    elif args.use_case == "Regression":
        for data, target_coords in train_loader:
            optimizer.zero_grad()

            outputs = model(data)
            outputs = outputs.view(-1, int(dataset.coords.shape[1]/3), 3)
            target_coords = target_coords.view(-1, int(dataset.coords.shape[1]/3), 3)
            
            lossChamfer, _ = chamfer_distance(outputs, target_coords)
            running_train_loss += lossChamfer.item()

            # Backward pass and optimizing
            lossChamfer.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        if args.use_case == "Classification":
            for data, target_label in val_loader:
                
                output_logits = model(data)
                pred_probs = F.softmax(output_logits, dim=1)

                loss = criterion_CEL(output_logits, target_label.long())
                running_val_loss += loss.item()

                # avg_val_loss[epoch] = loss.item()
        
        elif args.use_case == "Regression":
            for data, target_coords in val_loader:
                outputs = model(data)

                outputs = outputs.view(-1, int(dataset.coords.shape[1]/3), 3)  
                target_coords = target_coords.view(-1, int(dataset.coords.shape[1]/3), 3)
                lossChamfer, _ = chamfer_distance(outputs, target_coords)

                avg_val_loss[epoch] = lossChamfer.item()
    
    avg_train_loss[epoch] = running_train_loss / num_train_batches
    avg_val_loss[epoch] = running_val_loss / num_val_batches
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    
    if args.use_case == "Classification":
        print(f"\nEpoch {epoch+1}/{args.num_epochs} completed in {epoch_duration:.2f} seconds with a training error of {avg_train_loss[epoch]:.4f} a validation loss of {avg_val_loss[epoch]:.4f}\n")
        logging.info(f"\nEpoch {epoch+1}/{args.num_epochs} completed in {epoch_duration:.2f} seconds with a training error of {avg_train_loss[epoch]:.4f} a validation loss of {avg_val_loss[epoch]:.4f}\n")
    elif args.use_case == "Regression":
        print(f"\nEpoch {epoch+1}/{args.num_epochs} completed in {epoch_duration:.2f} seconds with a training error of {avg_train_loss[epoch]:.4f}, a validation loss of {avg_val_loss[epoch]:.4f} and a relative training error of {avg_train_loss[epoch]/dataset.dist_max:.4f}\n")
        logging.info(f"\nEpoch {epoch+1}/{args.num_epochs} completed in {epoch_duration:.2f} seconds with a training error of {avg_train_loss[epoch]:.4f}, a validation loss of {avg_val_loss[epoch]:.4f} and a relative training error of {avg_train_loss[epoch]/dataset.dist_max:.4f}\n")

    # Check Early Stopping
    if early_stopping(avg_val_loss[epoch]):
        break   # Stop training

print("Entrenamiento completado\n")
logging.info("Entrenamiento completado\n")

# Plot training and validation errors
plt.figure(figsize=(10, 6))
epochs_range = range(1, len(avg_train_loss[:epoch+1]) + 1)

plt.plot(epochs_range, avg_train_loss[:epoch+1], label='Training Loss')
plt.plot(epochs_range, avg_val_loss[:epoch+1], label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
os.makedirs(os.path.dirname(f"{os.getcwd()}/Models/"), exist_ok=True)
plot_path = Path("Models/loss_plot.png")
plt.savefig(plot_path)
plt.close()

print(f"Loss plot saved at {plot_path}\n")
logging.info(f"Loss plot saved at {plot_path}\n")

# Model saving
torch.save(model.state_dict(), f"Models/Model_{epoch}_ep.pth")
 
print("Model saved\n")
logging.info("Model saved\n")

# Test evaluation
model.eval()
aver=time.time()

all_preds = []
all_targets = []
total_correct = 0
total_samples = 0
total_loss = 0.0

with torch.no_grad():
    if args.use_case == "Classification":
        for data, target_label in test_loader:
            
            output_logits = model(data)

            pred_probs = F.softmax(output_logits, dim=1)
            correct_predictions = (pred_probs.argmax(dim=1) == target_label).sum().item()

            preds = pred_probs.argmax(dim=1)
            # print(f"length pred_probs: {len(pred_probs)}, length correct_predictions: {correct_predictions}, has to be equal to {pred_probs.argmax(dim=1) == target_label}")
            # accuracy = correct_predictions / len(target_label) * 100

            # --- ACCUMULATE METRICS (Don't print yet) ---
            total_correct += (preds == target_label).sum().item()
            total_samples += len(target_label)
            
            # Store predictions/targets for Confusion Matrix later
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target_label.cpu().numpy())

        avg_accuracy = (total_correct / total_samples) * 100

        print(f"Test Accuracy: {avg_accuracy:.2f}%")
        print(f"Total test samples: {total_samples}")
        print(f"Correctly predicted samples: {total_correct}")
        
        logging.info(f"Test Accuracy: {avg_accuracy:.2f}%")
        logging.info(f"Total test samples: {total_samples}")
        logging.info(f"Correctly predicted samples: {total_correct}")

        # --- 3. GENERATE CONFUSION MATRIX (Once for whole dataset) ---
        # Convert lists back to tensors for your metric function
        preds_tensor = torch.tensor(all_preds)
        targets_tensor = torch.tensor(all_targets)
        
        cm = multiclass_confusion_matrix(
            preds_tensor, 
            targets_tensor, 
            num_classes=dataset.manifest['label_idx'].nunique()
        )
        
        # Plotting (Same as before)
        cm = cm.numpy() # It's already on CPU because we created tensors from lists
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Handle labels
        # Note: Ensure dataset.manifest is accessible here
        unique_labels = sorted(dataset.manifest['label_idx'].unique()) 
        plt.xticks(np.arange(len(unique_labels)), unique_labels, rotation=45)
        plt.yticks(np.arange(len(unique_labels)), unique_labels)
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        os.makedirs("Models", exist_ok=True)
        plt.savefig(Path("Models/confusion_matrix.png"))
        plt.close()
        print(f"Confusion matrix saved at Models/confusion_matrix.png")
        # logging.info(f"Test Accuracy: {accuracy:.2f}%")
        # print(f"Test Accuracy: {accuracy:.2f}%")

        # print(f"Total test samples: {len(target_label)}")
        # print(f"Correctly predicted samples: {correct_predictions}")
        # logging.info(f"Total test samples: {len(target_label)}")
        # logging.info(f"Correctly predicted samples: {correct_predictions}")

        # # Compute confusion matrix
        # cm = multiclass_confusion_matrix(pred_probs.argmax(dim=1), target_label.long(), num_classes=dataset.manifest['label_idx'].nunique())
        # cm = cm.cpu().numpy()
        # # Save confusion matrix as a heatmap
        # plt.figure(figsize=(8, 6))
        # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.title('Confusion Matrix')
        # plt.colorbar()
        # unique_labels = dataset.manifest['label_idx'].unique()
        # plt.xticks(np.arange(len(unique_labels)), unique_labels, rotation=45)
        # plt.yticks(np.arange(len(unique_labels)), unique_labels)
        # plt.xlabel('Predicted Label')
        # plt.ylabel('True Label')
        # plt.tight_layout()
        # plt.savefig(Path("Models/confusion_matrix.png"))
        # plt.close()
        # print(f"Confusion matrix saved at Models/confusion_matrix.png")

    elif args.use_case == "Regression":
        for data, target_coords in test_loader:
            
            outputs = model(data)

            outputs = outputs.view(-1, int(dataset.coords.shape[1]/3), 3)  
            target_coords = target_coords.view(-1, int(dataset.coords.shape[1]/3), 3)
            lossChamfer, _ = chamfer_distance(outputs, target_coords)
            
            test_loss = lossChamfer.item()


# aver2=time.time()
# print(f"Tiempo por sample: {(aver2-aver)/len(test_dataset):.6f} seconds")

# print("Error medio de test: ", test_loss.item())
# logging.info(f"Error medio de test: {test_loss.item()}")

print("Fin de la ejecución")
logging.info("Fin de la ejecución")

# python CNN_general.py -i /home/newfasant/N101-IA/Datasets/Reorganized/Classification_6000_0_16_f_16_d_POV_90.0_SNR_10.0 -u Classification -d ISAR -e 100