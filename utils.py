import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import seaborn as sns
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

class TrainDataManager():
    def __init__(self, dataset_dir, batch_size, norm):
        self.dataset_dir = dataset_dir
        self.batch_size  = batch_size
        self.norm        = norm
                
    def Load_Dataset(self):        
        X_train, Y_train = np.load(self.dataset_dir+'x_train.npy'), np.load(self.dataset_dir+'y_train.npy')
        
        if self.norm == True:
            X_train_max, X_train_min= X_train.max(), X_train.min()
            X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
    
        Y_train = np.reshape(Y_train, (-1, 1))            
        
        print(f'The shape of X_TRAIN: {np.shape(X_train)} | The shape of Y_TRAIN: {np.shape(Y_train)}')
            
        train_data = Time_Series_dataset(X_train, Y_train)
    
        return train_data
    
    def Load_DataLoader(self, train):
        return DataLoader(train, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True)
                

class TestDataManager():
    def __init__(self, dataset_dir, batch_size, norm):
        self.dataset_dir = dataset_dir
        self.batch_size  = batch_size
        self.norm        = norm
            
    def Load_Dataset(self):        
        X_test,  Y_test  = np.load(self.dataset_dir+'x_test.npy') , np.load(self.dataset_dir+'y_test.npy')
        
        if self.norm == True:    
            X_test_max, X_test_min= X_test.max(), X_test.min()
            X_test  = (X_test - X_test_min) / (X_test_max - X_test_min)
        
        Y_test  = np.reshape(Y_test, (-1, 1))
        
        print(f'The shape of X_TEST : {np.shape(X_test)}  | The shape of Y_TEST : {np.shape(Y_test)}')
        
        test_data  = Time_Series_dataset(X_test , Y_test )

        return test_data
    
    def Load_DataLoader(self, test):
        return DataLoader(test, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False)

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