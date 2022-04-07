import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.ResNet import *
from utils import *

def fgsm_attack(signal, epsilon, gradient):
    sign_gradient = gradient.sign()
    perturbed_signal = signal + epsilon * sign_gradient
    perturbed_sginal = torch.clamp(perturbed_signal, 0, 1)
    return perturbed_signal

DEVICE = 'cuda'
BATCH_SIZE = 256

DATASET_DIR                 = f'./dataset/seizure/'
DM                          = DataManager(DATASET_DIR, BATCH_SIZE)
TRAIN_DATA, TEST_DATA       = DM.Load_Dataset()
TRAIN_LOADER, TEST_LOADER   = DM.Load_DataLoader(TRAIN_DATA, TEST_DATA)

model = ResNet(input_size = 256,                      
               input_channel = 18,    
               num_label = 2).to(DEVICE)
model.load_state_dict(torch.load('./saved_models/ResNet/seizure_0.77.pth'))
criterion = nn.CrossEntropyLoss().to(DEVICE)
count = 0
for idx, batch in enumerate(TEST_LOADER):
    X, Y = batch
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)
    X.requires_grad_(True)
    y_pred  = model(X)
    Y_train = Y.squeeze(-1)
    loss    = criterion(y_pred, Y_train)

    model.zero_grad()
    loss.backward()

    gradient = X.grad.data
    epsilon        = 0.5
    preturbed_data = fgsm_attack(X, epsilon, gradient)

    attackOutput = model(preturbed_data)
    if (torch.argmax(y_pred) != torch.argmax(attackOutput)):
        count += 1
print(count)
    