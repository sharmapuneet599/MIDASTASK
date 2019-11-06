import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
from tqdm.notebook import tqdm
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torchvision
from torch import optim
inputPath = input()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.Convolution1 = nn.Sequential(
            
            # convolutional layer
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            
            # Batch Normalization layer
            nn.BatchNorm2d(64),
            
            # Activation Function
            nn.ReLU(),
            
            nn.Dropout(p=0.2),
            
            # max pooling layer
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            
            nn.BatchNorm2d(128),
            
            nn.ReLU(),
            
            nn.Dropout(p=0.2),
            
            nn.MaxPool2d(kernel_size=2),
            
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            
            nn.BatchNorm2d(256),
            
            nn.ReLU()
        
          
            )
                
        #Fully Connected 1
        
        self.fc = nn.Linear(10*35*256, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 5)


    def forward(self, x):
        
        output = self.Convolution1(x)
                
        output = output.view(output.size(0), -1)
        
        output = self.fc(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

# We will train model on GPU. So we need to move the model first to the GPU.
model = CNN()
if torch.cuda.is_available():
    model.cuda()
model.eval()
model.load_state_dict(torch.load(r'C:\Users\Tarun\Desktop\MIDAS project\MY_MODEL1.pt'))

path = os.path.normpath(inputPath)
count = 0
for i in os.listdir(path):
    audio, sr = librosa.load(os.path.join(path,i))
    #fixing lenght and processing audio file
    y = librosa.util.fix_length(audio, 66150)
    S = np.abs(librosa.stft(y))
    D_short = librosa.feature.chroma_stft(S=S, n_chroma=40)

    #applying transformations on the data
    D_short = (D_short-D_short.min())/(D_short.max()-D_short.min())
    trans = transforms.ToPILImage()
    a = trans(D_short)
    trans = transforms.Resize((40,140))
    a = trans(a)
    trans = transforms.ToTensor()
    a = trans(a)
    a= a.unsqueeze(0)
    count+=1
    
    a = a.to('cuda')
    output = model(a)
    result = torch.argmax(output)
    if(result == 0):
        res = "a"
    elif(result == 1):
        res = "b"
    elif(result == 2):
        res = "c"
    elif(result == 3):
        res = "d"
    elif(result == 4):
        res = "e"
    with open('myfile.txt', 'a') as f_out:
        f_out.write(i)
        f_out.write(',')
        f_out.write(res)
        f_out.write('\n')
f_out.close()
