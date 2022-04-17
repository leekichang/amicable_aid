import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import sklearn.metrics as metrics
import warnings
import argparse
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from dataloader import *
from models import *

warnings.filterwarnings(action = 'ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='train and test')
    
    parser.add_argument('--dataset'   , default = 0 , type = int,
                        choices=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument('--testset'   , default = 0 , type = int,
                        choices=[0, 1, 2])
    parser.add_argument('--model'     , default ='ResNet', type = str,
                        choices=['ResNet', 'MaDNN', 'MaCNN'])
    parser.add_argument('--batch_size', default = 256 , type = int)
    parser.add_argument('--norm'      , default = True, type = bool)
    args = parser.parse_args()
    return args

args = parse_args()

if args.testset == 0:
    test_models = ['nexus4']
elif args.testset == 1:
    test_models = ['s3']
elif args.testset == 2:
    test_models = ['s3mini']


if args.dataset == 0:
    models = ['nexus4']
    save_model_name = f'{args.model}_nexus4'
elif args.dataset == 1:
    models = ['s3']
    save_model_name = f'{args.model}_s3'
elif args.dataset == 2:
    models = ['s3mini']
    save_model_name = f'{args.model}_s3mini'
elif args.dataset == 3:
    models = ['nexus4', 's3']
    save_model_name = f'{args.model}_nexus4+s3'
elif args.dataset == 4:
    models = ['nexus4', 's3mini']
    save_model_name = f'{args.model}_nexus4+s3mini'
elif args.dataset == 5:
    models = ['s3', 's3mini']
    save_model_name = f'{args.model}_s3+s3mini'
elif args.dataset == 6:
    models = ['nexus4', 's3', 's3mini']
    save_model_name = f'{args.model}_nexus4+s3+s3mini'
    

BATCH_SIZE   = args.batch_size
DATASET_DIR  = './dataset/'
DEVICE       = 'cuda' if torch.cuda.is_available() == True else 'cpu'

TEST_DM      = DataManager(DATASET_DIR, BATCH_SIZE, models = test_models, train=False, norm = args.norm)
TEST_DATA    = TEST_DM.Load_Dataset()
TEST_LOADER  = TEST_DM.Load_DataLoader(TEST_DATA)

print(f"WORKING WITH {DEVICE}")
norm = args.norm
model_save_path = "./saved_models/" + save_model_name + "/" if norm == False else "./saved_models/" + save_model_name + "/norm1/"

if args.model == 'ResNet':
    model = ResNet(input_size    = 256,                      
                   input_channel = 6,    
                   num_label     = 6).to(DEVICE)
elif args.model == 'MaDNN':
    model = MaDNN(input_size    = 256,                      
                  input_channel = 6,    
                  num_label     = 6).to(DEVICE)
elif args.model == 'MaCNN':
    model = MaCNN(input_size    = 256,                      
                  input_channel = 6,    
                  num_label     = 6,
                  sensor_num    = 2).to(DEVICE)

model.load_state_dict(torch.load(f'{model_save_path}{os.listdir(model_save_path)[-1]}'))
print(f'{model_save_path}{os.listdir(model_save_path)[-1]}')
model.to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)

with torch.no_grad():
    model.eval()
    Result_pred_val, Result_anno_val = [], []
    for idx, batch in enumerate(TEST_LOADER):
        X_val, Y_val = batch
        X_val, Y_val = X_val.to(DEVICE), Y_val.to(DEVICE)
        Y_pred_val = model(X_val)
        Y_val      = Y_val.squeeze(-1)
        
        LOSS_val = criterion(Y_pred_val, Y_val)
        Y_pred_val_np  = Y_pred_val.to('cpu').detach().numpy()
        Y_pred_val_np  = np.argmax(Y_pred_val_np, axis=1).squeeze()
        Y_val_np       = Y_val.to('cpu').detach().numpy().reshape(-1, 1).squeeze()  
        
        Result_pred_val = np.hstack((Result_pred_val, Y_pred_val_np))
        Result_anno_val = np.hstack((Result_anno_val, Y_val_np))
        
    Result_pred_np = np.array(Result_pred_val)
    Result_anno_np = np.array(Result_anno_val)
    Result_pred_np = np.reshape(Result_pred_np, (-1, 1))
    Result_anno_np = np.reshape(Result_anno_np, (-1, 1))


get_metrics(pred    = Result_pred_np,
            anno    = Result_anno_np,
            n_label = 6)

perf = metrics.accuracy_score(Result_anno_np, Result_pred_np)
txt_save_path =  "./results/" + save_model_name + "/" if norm == False else "./results/" + save_model_name + "/norm1/"

if not os.path.isdir("./results/" + save_model_name + "/"):
    os.mkdir("./results/" + save_model_name + "/")
if not os.path.isdir("./results/" + save_model_name + "/norm1/"):
    os.mkdir("./results/" + save_model_name + "/norm1/")
    
f    = open(f'{txt_save_path}.txt', 'w')
f.write(f'{perf*100:.2f}')
f.close()