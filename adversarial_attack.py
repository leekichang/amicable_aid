# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.ResNet import *
from models.mann import *
from models.laxcat import *
from models.dual import *
from utils import *
from config import *
import argparse
import sys


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

def parse_args():
    parser = argparse.ArgumentParser(description='train and test')
    
    parser.add_argument('--dataset'   , default = 'keti' , type = str,
                        choices=['motion', 'seizure', 'wifi', 'keti', 'PAMAP2'])
    parser.add_argument('--model'     , default ='ResNet', type = str,
                        choices=['ResNet', 'MaCNN', 'MaDNN', 'LaxCat', 'RFNet'])
    parser.add_argument('--aid'          , default =True, type = str,
                        choices=['True', 'False'])
    args = parser.parse_args()
    return args

args = parse_args()

DEVICE = 'cuda'
norm = False

save_model_name = args.model
model_save_path = "./saved_models/" + save_model_name + "/"

model_path = model_save_path if norm == False else model_save_path+'/normalized/'
model_name = [file for file in os.listdir(model_path) if file.endswith('.pth') and args.dataset in file][-1]

print(f'Model Name : {model_name}', file=sys.stderr)

DATASET_DIR                 = f'./dataset/{args.dataset}/'
DM                          = TestDataManager(DATASET_DIR, BATCH_SIZE, norm)
TEST_DATA                   = DM.Load_Dataset()
TEST_LOADER                 = DM.Load_DataLoader(TEST_DATA)

model_config = dataset_config[args.dataset]

if args.model == 'ResNet':
    model = ResNet(input_size = 256,                      
                   input_channel = model_config['n_channel'],    
                   num_label = model_config['n_label']).to(DEVICE)
elif args.model == 'MaCNN':
    model = MaCNN(input_size    = input_size,
                  input_channel = model_config['n_channel'],
                  num_label     = model_config['n_label'  ], 
                  sensor_num    = int(model_config['n_channel'] / model_config['n_axis'])).to(DEVICE)
elif args.model == 'MaDNN':
    model = MaDNN(input_size    = input_size,
                  input_channel = model_config['n_channel'],
                  num_label     = model_config['n_label'  ]).to(DEVICE)
elif args.model == 'LaxCat':
    model = LaxCat(input_size    = input_size,
                   input_channel = model_config['n_channel'],
                   num_label     = model_config['n_label'  ]).to(DEVICE)
elif args.model == 'RFNet':
    model = RFNet( win_len       = input_size,
                   input_channel = model_config['n_channel'],
                   num_classes   = model_config['n_label'  ]).to(DEVICE)

model.load_state_dict(torch.load(model_path+model_name))

original_model = model
reference_model = model

criterion = nn.CrossEntropyLoss().to(DEVICE)

if args.aid == 'True':
    aid = True
else:
    aid = False

alpha_dict = {'keti' : 0.02, 'wifi': 0.001, 'seizure' : 0.08, 'PAMAP2':0.004, 'motion':1}

alpha   = alpha_dict[args.dataset]
eps     = 50
ACCS    = []
epsilons = [alpha*e for e in range(eps+1)]

print(f'Model : {model_name} | Dataset : {args.dataset} | Aid = {aid}', file=sys.stderr)



for idx, e in enumerate(range(eps+1)):
    Result_pred_val, Result_anno_val = [], []
    for jdx, batch in enumerate(TEST_LOADER):
        X_val, Y_val = batch
        X_val, Y_val = X_val.to(DEVICE), Y_val.to(DEVICE)
        
        X_val.requires_grad_(True)
        
        model.load_state_dict(torch.load(model_path+model_name))
        original_model.load_state_dict(torch.load(model_path+model_name))
        model.train()
        Y_pred_val = model(X_val)
        Y_val      = Y_val.squeeze(-1)

        LOSS_val = criterion(Y_pred_val, Y_val)
        
        model.zero_grad()
        LOSS_val.backward()
        
        gradient = X_val.grad.data
        epsilon  = alpha*e
        
        perturbed_data = fgsm(X_val, epsilon, gradient, aid=aid)
        
        with torch.no_grad():
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
    print(f'Epsilon : {epsilon : >3.4f}, accuracy : {ACC_VAL*100:.4f}', file = sys.stdout)
    
