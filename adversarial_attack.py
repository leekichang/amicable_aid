import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.ResNet import *
from utils import *
from config import *

def fgsm(signal, epsilon, gradient, aid):
    if epsilon == 0:
        return signal
    if aid == True:
        aid = -1
    else:
        aid = 1
    perturbation = torch.mul(gradient.sign(), aid*epsilon)
    perturbed_signal = signal + perturbation
    return perturbed_signal

DEVICE = 'cuda'
dataset = 'keti'
norm = False

model_path = model_save_path if norm == False else model_save_path+'/normalized/'
model_name = [file for file in os.listdir(model_path) if file.endswith('.pth') and dataset in file][0]

print(f'Model Name : {model_name}')

DATASET_DIR                 = f'./dataset/{dataset}/'
DM                          = TestDataManager(DATASET_DIR, BATCH_SIZE, norm)
TEST_DATA                   = DM.Load_Dataset()
TEST_LOADER                 = DM.Load_DataLoader(TEST_DATA)

model = ResNet(input_size = 256,                      
               input_channel = dataset_config[dataset]['n_channel'],    
               num_label = dataset_config[dataset]['n_label']).to(DEVICE)

model.load_state_dict(torch.load(model_path+model_name))

original_model = model

criterion = nn.CrossEntropyLoss().to(DEVICE)

eps     = 50
aid     = False
ACCS    = []
alpha   = 1
epsilons = [alpha*e for e in range(eps+1)]
print(f'Model : {model_name} | Dataset : {dataset} | Aid = {aid}')

for idx, e in enumerate(range(eps+1)):
    print(f'testing : {e}')
    Result_pred_val, Result_anno_val = [], []
    for jdx, batch in enumerate(TEST_LOADER):
        model = original_model
        X_val, Y_val = batch
        X_val, Y_val = X_val.to(DEVICE), Y_val.to(DEVICE)
        
        X_val.requires_grad_(True)
        
        model.eval()
        Y_pred_val = model(X_val)
        Y_val      = Y_val.squeeze(-1)

        LOSS_val = criterion(Y_pred_val, Y_val)
        
        model.zero_grad()
        LOSS_val.backward()
        
        gradient = X_val.grad.data
        epsilon  = alpha*e
        
        perturbed_data = fgsm(X_val, epsilon, gradient, aid=aid)
        
        original_model.eval()
        attackOutput = original_model(perturbed_data)
                
        Y_pred_val_np  = attackOutput.to('cpu').detach().numpy()
        Y_pred_val_np  = np.argmax(Y_pred_val_np, axis=1).squeeze()
        Y_val_np       = Y_val.to('cpu').detach().numpy().reshape(-1, 1).squeeze() 

        Result_pred_val = np.hstack((Result_pred_val, Y_pred_val_np))
        Result_anno_val = np.hstack((Result_anno_val, Y_val_np))
    
    Result_pred_np = np.array(Result_pred_val)
    Result_anno_np = np.array(Result_anno_val)
    Result_pred_np = np.reshape(Result_pred_np, (-1, 1))
    Result_anno_np = np.reshape(Result_anno_np, (-1, 1))
    
    ACC_VAL        = metrics.accuracy_score(Result_anno_np, Result_pred_np)
    
    ACCS.append(ACC_VAL)
    
for e, acc in enumerate(ACCS):
    print(f'Epsilon : {epsilons[e] : >3.4f}, accuracy : {acc*100:.4f}')