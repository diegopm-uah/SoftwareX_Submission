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
from pytorch3d.loss import chamfer_distance
import csv
import time
import argparse
from datetime import datetime
from torcheval.metrics.functional import multiclass_confusion_matrix
import json

seed = 10

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

userPath = os.getcwd().split('/')[2]
cuda_own = "cuda:0"
if userPath == "newfasant2":
    userPath = userPath + "/N101"
    cuda_own = "cuda:1"

device = (
    cuda_own
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
         )

parser = argparse.ArgumentParser(description='Procesar argumentos para enviarlos a los distintos programas necesarios.')
parser.add_argument('-i', '--input_path', type=str, help='Ruta de la carpeta en la que se encuentra el dataset (terminada en una carpeta formato Classification_660_0_64_f_64_d).')
parser.add_argument('-u', '--use_case', type=str, default='Classification', choices=['Classification', 'Regression'], help='Tipo de uso que deseado  a la red')
# Below, npy refers to the field matrix, fft refers to the FFT of the field matrix and ISAR refers to the ISAR images generated from the field matrix.
parser.add_argument('-d', '--data_type', type=str, default='ISAR', choices=['npy','fft','ISAR','fft_perfil','perfil_png','field','field_amp_npy','field_ph_npy','field_amp_png','field_ph_png'], help='Data type to be used to train the network.')

args = parser.parse_args()

class CustomDataset(Dataset):
    """Custom dataset for loading data files and their corresponding labels.

    Initializes the CustomDataset instance.

    Args:
        npy_dir (str): Directory containing the data files.
        labels_file (str): File path to the file containing labels.
        transform_ISAR (callable, optional): Optional transform to be applied on a sample for ISAR data type.
        transform_npy (callable, optional): Optional transform to be applied on a sample for npy data type.
    """

    def __init__(self, dir, transform_ISAR=None, transform_npy=None):

        self.dir = dir

        if args.data_type == 'npy':
            # List all .npy files in the directory.
            self.file_names = [f for f in os.listdir(dir) if f.endswith('.npy') and f.startswith('sample_') and not f.endswith('fft.npy') and not f.endswith('perfil.npy') and not f.endswith('field.npy') and not f.endswith('field_amp.npy') and not f.endswith('field_ph.npy')]
            self.transform = transform_npy
        elif args.data_type == 'fft':
            # List all fft files in the directory.
            self.file_names = [f for f in os.listdir(dir) if f.endswith('fft.npy')]
            self.transform = transform_npy
        elif args.data_type == 'fft_perfil':
            # List all fft (in profiles) files in the directory.
            self.file_names = [f for f in os.listdir(dir) if f.endswith('perfil.npy')]
            self.transform = transform_npy
        elif args.data_type == 'field':
            # List all field files in the directory.
            self.file_names = [f for f in os.listdir(dir) if f.endswith('field.npy')]
            self.transform = transform_npy
        elif args.data_type == 'field_amp_npy':
            # List all files containing the amplitude of the field in the directory.
            self.file_names = [f for f in os.listdir(dir) if f.endswith('field_amp.npy')]
            self.transform = transform_npy
        elif args.data_type == 'field_ph_npy':
            # List all files containing the complex phase of the field in the directory.
            self.file_names = [f for f in os.listdir(dir) if f.endswith('field_ph.npy')]
            self.transform = transform_npy
        elif args.data_type == 'perfil_png':
            # List all the images plotting the fft (in profiles) in the directory.
            self.file_names = [f for f in os.listdir(dir) if f.endswith('perfil.png')]
            self.transform = transform_ISAR
        elif args.data_type == 'field_amp_png':
            # List all the images plotting the amplitude of the field in the directory.
            self.file_names = [f for f in os.listdir(dir) if f.endswith('amp.png')]
            self.transform = transform_ISAR
        elif args.data_type == 'field_ph_png':
            # List all the images plotting the complex phase of the field in the directory.
            self.file_names = [f for f in os.listdir(dir) if f.endswith('ph.png')]
            self.transform = transform_ISAR
        elif args.data_type == 'ISAR':
            # List all the ISAR images in the directory.
            self.file_names = [f for f in os.listdir(dir) if f.endswith('.png') and not f.endswith('perfil.png') and not f.endswith('amp.png') and not f.endswith('ph.png')]
            self.transform = transform_ISAR
            
        # The network must say which of the geometries corresponds to a particular input
        _, _, result = dir.split('/')[-1].partition('_')  # Divides in 3 parts: before, splitter and after
        result = f"_{result}"
        # Load labels vector from the labels file.
        self.labels_vector = np.load(dir + '/labels_vector' + result + '.npy')
            
        # Sort files based on the sample id which is the number right after "sample_". For example, "sample_5.npy" will be split into ['sample', '5', '.npy'].
        self.file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            
    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.file_names)
    
    def __getitem__(self, idx):

        # Construct the full file path for the sample.
        file_path = os.path.join(self.dir, self.file_names[idx])
        
        if args.data_type == 'npy' or args.data_type == 'fft' or args.data_type == 'fft_perfil' or args.data_type == 'field' or args.data_type == 'field_amp_npy' or args.data_type == 'field_ph_npy':
            # Load the numpy array with shape (2 x n x m).
            data = np.load(file_path)
            data = np.transpose(data, (2, 0, 1))
            
            # Convert the numpy array to a PyTorch tensor.
            data = torch.tensor(data, dtype=torch.float32).to(device)
        elif args.data_type == 'ISAR' or args.data_type == 'perfil_png' or args.data_type == 'field_amp_png' or args.data_type == 'field_ph_png':
            # Load the image and convert it to "L"
            data = Image.open(file_path).convert("L")
            
            # If a transform is set, apply it
            if self.transform:
                data = self.transform(data).to(device)
            else:
                data.to(device)

        target = torch.tensor(self.labels_vector[idx], dtype=torch.float32).to(device)
        
        return data, target
    
    def dist_max_calc(self):
        # Returns maximux size of the coordinates vector of the points
        sample_coords = self.coords[1].reshape(int(self.coords.shape[1]/3),3)
        dist_max = 0.0
        for i in range(len(sample_coords)):
            if sum(sample_coords[i] ** 2) > dist_max:
                dist_max = sum(sample_coords[i] ** 2)
            dist_max = np.sqrt(dist_max)
        return dist_max

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
        x = self.Flatten(x)
        return x

class GeometryPredictor(nn.Module):

    def __init__(self, num_labels=None, coords_width=None):

        super(GeometryPredictor, self).__init__()
        
        self.cnn = CNNModule()

        # Fully connected layers
        self.fc1 = nn.LazyLinear(4096)  # Adjust size if needed.
        self.fc2 = nn.LazyLinear(4096)  #
        
        self.fc3 = nn.LazyLinear(num_labels)  # Number of outputs equal to num_labels.

    def forward(self, x):

        # Process the input through the CNN to extract features.
        features = self.cnn(x)

        # Pass the features through fully connected layers.
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        # Return the final results of the last layer of the network
        output = self.fc3(x)

        return output


transform_I = transforms.Compose([
    transforms.Resize((128, 128)),   # Change this depending of image sizes
    transforms.ToTensor()            # Converting images to tensors
])

dataset = CustomDataset(dir = args.input_path, transform_ISAR=transform_I)

num_prev_clas = 4

model = GeometryPredictor(num_labels=num_prev_clas).to(device)
model.load_state_dict(torch.load('/home/newfasant2/N101/N101-IA/CNN/Models/Classification_field_ph_npy_AMSS_POV_50.0_SNR_10.0_2000_0samples_15ep_16bs.pth',weights_only=False, map_location=torch.device('cpu')))
model.eval()

train_size = int(0.99 * len(dataset))  # 70% for train
val_size = int(0.001 * len(dataset))  # 15% for validation
test_size = len(dataset) - train_size - val_size  # 15% for test

generator = torch.Generator().manual_seed(seed)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

def disc_func(probs, u_class, t):
    # print(probs[0])
    if max(probs[0]) > t:
        return np.argmax(probs[0].cpu())
    else:
        return u_class

for t_aux in range(26):
    suc = np.empty(len(train_dataset))
    ind = 0
    t = t_aux / 25
    with torch.no_grad():
        for data, target_label in train_loader:
            output_logits = model(data)
            pred_probs = abs(F.softmax(output_logits, dim=1))
            pred_probs = pred_probs / (pred_probs.sum())
            result = disc_func(pred_probs, num_prev_clas, t)
            suc[ind] = result
            ind += 1
        # print(suc)
    print(f"Para t={t}, el porcentaje de no reconocidas es:", 100 - 100 * len(suc[suc!=num_prev_clas])/len(suc), "%")
