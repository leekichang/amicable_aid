import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import sklearn.metrics as metrics
import argparse
import warnings
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from dataloader import *
from models import *

warnings.filterwarnings(action = 'ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='train and test')
    
    parser.add_argument('--dataset'   , default = 0 , type = int,
                        choices=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument('--model'     , default ='ResNet', type = str,
                        choices=['ResNet', 'MaDNN', 'MaCNN'])    
    parser.add_argument('--norm'      , default = True  , type = bool  )
    parser.add_argument('--epochs'    , default = 50    , type = int  )
    parser.add_argument('--lr'        , default = 1e-3  , type = float)
    parser.add_argument('--batch_size', default = 256    , type = int  )
    args = parser.parse_args()
    return args

args = parse_args()

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

BATCH_SIZE  , EPOCH, LR    = args.batch_size, args.epochs, args.lr
DATASET_DIR                = f'./dataset/'
DEVICE                     = 'cuda' if torch.cuda.is_available() == True else 'cpu'
TRAIN_DM    , TEST_DM      = DataManager(DATASET_DIR, BATCH_SIZE, models = models, train = True, norm = True),\
                           DataManager(DATASET_DIR, BATCH_SIZE, models = models, train=False, norm = True)
TRAIN_DATA  , TEST_DATA    = TRAIN_DM.Load_Dataset(), TEST_DM.Load_Dataset()
TRAIN_LOADER, TEST_LOADER  = TRAIN_DM.Load_DataLoader(TRAIN_DATA), TEST_DM.Load_DataLoader(TEST_DATA)

print(f"WORKING WITH {DEVICE}")
print(f"{save_model_name}")
norm = args.norm
model_save_path = "./saved_models/" + save_model_name + "/" if norm == False else "./saved_models/" + save_model_name + "/norm1/"
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

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

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

SOTA_ACC_VAL      , SOTA_LOSS_VAL      = 0 , 0
bestResult_pred_np, bestResult_anno_np = [], []
bestModel                              = model

for epoch in tqdm(enumerate(range(EPOCH)), desc="EPOCHS"):
    model.train()
    LOSS_TRACE_FOR_TRAIN, LOSS_TRACE_FOR_VAL = [], []
    for idx, batch in enumerate(TRAIN_LOADER):
        optimizer.zero_grad()
        X_train, Y_train = batch
        X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)

        Y_pred_train = model(X_train)
        Y_train = Y_train.squeeze(-1)

        LOSS_train = criterion(Y_pred_train, Y_train)

        LOSS_TRACE_FOR_TRAIN.append(LOSS_train.cpu().detach().numpy())
        LOSS_train.backward()
        optimizer.step()
    
    with torch.no_grad():
        model.eval()
        Result_pred_val, Result_anno_val = [], []
        for idx, batch in enumerate(TEST_LOADER):
            X_val, Y_val = batch
            X_val, Y_val = X_val.to(DEVICE), Y_val.to(DEVICE)

            Y_pred_val = model(X_val)
            Y_val      = Y_val.squeeze(-1)

            LOSS_val = criterion(Y_pred_val, Y_val)
            LOSS_TRACE_FOR_VAL.append(LOSS_val.cpu().detach().numpy())

            Y_pred_val_np  = Y_pred_val.to('cpu').detach().numpy()
            Y_pred_val_np  = np.argmax(Y_pred_val_np, axis=1).squeeze()
            Y_val_np       = Y_val.to('cpu').detach().numpy().reshape(-1, 1).squeeze()     

            Result_pred_val = np.hstack((Result_pred_val, Y_pred_val_np))
            Result_anno_val = np.hstack((Result_anno_val, Y_val_np))
    
    Result_pred_np = np.array(Result_pred_val)
    Result_anno_np = np.array(Result_anno_val)
    Result_pred_np = np.reshape(Result_pred_np, (-1, 1))
    Result_anno_np = np.reshape(Result_anno_np, (-1, 1))
    
    ACC_VAL        = metrics.accuracy_score(Result_anno_np, Result_pred_np)
    AVG_LOSS_TRAIN = np.average(LOSS_TRACE_FOR_TRAIN)
    AVG_LOSS_VAL   = np.average(LOSS_TRACE_FOR_VAL)
    
    if ACC_VAL > SOTA_ACC_VAL:
        SOTA_ACC_VAL       = ACC_VAL
        SOTA_LOSS_VAL      = AVG_LOSS_VAL
        bestModel          = model
        bestResult_pred_np = Result_pred_np
        bestResult_anno_np = Result_anno_np
    
    printLearningData(epoch, EPOCH, AVG_LOSS_TRAIN, AVG_LOSS_VAL, ACC_VAL)
    
get_metrics(pred    = bestResult_pred_np,
            anno    = bestResult_anno_np,
            n_label = 6)

save_name = f'{model_save_path}{args.model}_{SOTA_ACC_VAL*100:.2f}.pth' if norm == False else f'{model_save_path}{args.model}_norm_{SOTA_ACC_VAL*100:.2f}.pth'
torch.save(bestModel.state_dict(), save_name)