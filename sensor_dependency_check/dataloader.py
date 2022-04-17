import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import seaborn as sns
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import sys

class DataManager():
    def __init__(self, dataset_dir, batch_size, models, norm, train):
        self.dataset_dir = dataset_dir
        self.batch_size  = batch_size
        self.models      = models
        self.train       = train
        self.norm        = norm
                
    def Load_Dataset(self):
        X_ = []
        Y_ = []
        if self.train == True:
            is_train = 'train'
        else:
            is_train = 'test'
        for model in self.models:
            X, Y = np.load(self.dataset_dir+f'x_{is_train}_{model}_256.npy'), np.load(self.dataset_dir+f'y_{is_train}_{model}_256.npy')
            print(X.shape, Y.shape)
            X_.append(X)
            Y_.append(Y)
        
        X_ = np.vstack(X_)
        Y_ = np.vstack(Y_)
        
        if self.norm == True:
            X_max, X_min= X_.max(axis=(0, 1)), X_.min(axis=(0, 1))
            X_ = (X_ - X_min) / (X_max - X_min)
    
        Y_ = np.reshape(Y_, (-1, 1))            
        
        print(f'The shape of X_: {np.shape(X_)} | The shape of Y_: {np.shape(Y_)}', file=sys.stderr)
            
        data = Time_Series_dataset(X_, Y_)
    
        return data
    
    def Load_DataLoader(self, data):
        if self.train == True:
            return DataLoader(data, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        else:
            return DataLoader(data, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False)
                

class Time_Series_dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        self.len = len(self.X)
        return self.len
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx,:]).float()
        Y = torch.tensor(self.Y[idx,:]).long()
        return X, Y

def printLearningData(epoch, EPOCH, AVG_LOSS_TRAIN, AVG_LOSS_VAL, ACC_VAL):
    print(f'(epoch {epoch[0]+1 : 03}/{EPOCH: 03}) | Training Loss : {AVG_LOSS_TRAIN:.5f} | ',
          f'Validation Loss :{AVG_LOSS_VAL:.5f} | Validation Accuracy : {ACC_VAL*100:.2f} %', sep = '')

def get_metrics(pred, anno, n_label, plot=False):
    print(np.shape(pred))
    print(np.shape(anno))
    print(metrics.accuracy_score(anno, pred))
    conf_mat = metrics.confusion_matrix(anno, pred)
    print(conf_mat)
    print(metrics.classification_report(anno, pred))

    if plot == True:
        plt.rcParams["figure.figsize"] = (n_label, n_label)
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.show()
        conf_mat_sum = np.sum(conf_mat, axis=1)
        conf_mat_sum = np.reshape(conf_mat_sum, (n_label, 1))
        sns.heatmap(conf_mat/conf_mat_sum, annot=True, fmt='.2%', cmap='Blues')