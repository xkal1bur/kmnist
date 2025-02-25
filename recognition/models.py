from django.db import models

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
# Create your models here.
# Definición del modelo
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128*6*6, 4000)
        self.fc2 = nn.Linear(4000, 3490)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.drop1(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

# Carga del modelo y pesos (se hace una sola vez)
model = Net()
model.load_state_dict(torch.load('modelo_kanji_1500000_completo_40.pth'))
model.eval()

# Diccionario de mapeo de índices a símbolos Kanji
array_k = np.load('unicodes_kanji.npy')
def transform(karray):
    karray2 = np.array([int(s.replace('U+', ''), 16) for s in karray])
    karray3 = np.vectorize(chr)(karray2)
    return dict(enumerate(karray3))
label_map = transform(array_k)

print(label_map)
