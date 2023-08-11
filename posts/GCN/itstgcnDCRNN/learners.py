import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

# torch
import torch
import torch.nn.functional as F
#import torch_geometric_temporal
from torch_geometric_temporal.nn.recurrent import DCRNN

# utils
import copy
#import time
#import pickle
#import itertools
#from tqdm import tqdm
#import warnings

# rpy2
#import rpy2
#import rpy2.robjects as ro 
from rpy2.robjects.vectors import FloatVector
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as rpyn
GNAR = importr('GNAR') # import GNAR 
#igraph = importr('igraph') # import igraph 
ebayesthresh = importr('EbayesThresh').ebayesthresh

from .utils import convert_train_dataset
from .utils import DatasetLoader

def make_Psi(T):
    W = np.zeros((T,T))
    for i in range(T):
        for j in range(T):
            if i==j :
                W[i,j] = 0
            elif np.abs(i-j) <= 1 : 
                W[i,j] = 1
    d = np.array(W.sum(axis=1))
    D = np.diag(d)
    L = np.array(np.diag(1/np.sqrt(d)) @ (D-W) @ np.diag(1/np.sqrt(d)))
    lamb, Psi = np.linalg.eigh(L)
    return Psi

def trim(f):
    f = np.array(f)
    if len(f.shape)==1: f = f.reshape(-1,1)
    T,N = f.shape
    Psi = make_Psi(T)
    fbar = Psi.T @ f # apply dft 
    fbar_threshed = np.stack([ebayesthresh(FloatVector(fbar[:,i])) for i in range(N)],axis=1)
    fhat = Psi @ fbar_threshed # inverse dft 
    return fhat

def update_from_freq_domain(signal, missing_index):
    signal = np.array(signal)
    T,N = signal.shape 
    signal_trimed = trim(signal)
    for i in range(N):
        try: 
            signal[missing_index[i],i] = signal_trimed[missing_index[i],i]
        except: 
            pass 
    return signal

    
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, filters, 1)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
class StgcnLearner:
    def __init__(self,train_dataset,dataset_name = None):
        self.train_dataset = train_dataset
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.dataset_name = str(train_dataset) if dataset_name is None else dataset_name
        self.mindex= getattr(self.train_dataset,'mindex',None)
        self.mrate_eachnode = getattr(self.train_dataset,'mrate_eachnode',0)
        self.mrate_total = getattr(self.train_dataset,'mrate_total',0)
        self.mtype = getattr(self.train_dataset,'mtype',None)
        self.interpolation_method = getattr(self.train_dataset,'interpolation_method',None)
        self.method = 'STGCN'
    def learn(self,filters=32,epoch=50):
        self.model = RecurrentGCN(node_features=self.lags, filters=filters)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model.train()
        for e in range(epoch):
            cost = 0
            for t, snapshot in enumerate(self.train_dataset):
                y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).reshape(-1)
                cost = cost + torch.mean((y_hat-snapshot.y)**2)
            cost = cost / (t+1)
            cost.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print('{}/{}'.format(e+1,epoch),end='\r')
        # recording HP
        self.nof_filters = filters
        self.epochs = epoch+1
    def __call__(self,dataset):
        X = torch.tensor(dataset.features).float()
        y = torch.tensor(dataset.targets).float()
        yhat = torch.stack([self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr) for snapshot in dataset]).detach().squeeze().float()
        return {'X':X, 'y':y, 'yhat':yhat} 

class ITStgcnLearner(StgcnLearner):
    def __init__(self,train_dataset,dataset_name = None):
        super().__init__(train_dataset)
        self.method = 'IT-STGCN'
    def learn(self,filters=32,epoch=50):
        self.model = RecurrentGCN(node_features=self.lags, filters=filters)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model.train()
        train_dataset_temp = copy.copy(self.train_dataset)
        for e in range(epoch):
            f,lags = convert_train_dataset(train_dataset_temp)
            f = update_from_freq_domain(f,self.mindex)
            T,N = f.shape 
            data_dict_temp = {
                'edges':self.train_dataset.edge_index.T.tolist(), 
                'node_ids':{'node'+str(i):i for i in range(N)}, 
                'FX':f
            }
            train_dataset_temp = DatasetLoader(data_dict_temp).get_dataset(lags=self.lags)  
            cost = 0
            for t, snapshot in enumerate(train_dataset_temp):
                y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).reshape(-1)
                cost = cost + torch.mean((y_hat-snapshot.y)**2)
            cost = cost / (t+1)
            cost.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print('{}/{}'.format(e+1,epoch),end='\r')
        # record
        self.nof_filters = filters
        self.epochs = epoch+1

class GNARLearner(StgcnLearner):
    def __init__(self,train_dataset,dataset_name = None):
        super().__init__(train_dataset)
        self.method = 'GNAR'
    def learn(self):
    
        self.N = np.array(self.train_dataset.features).shape[1]
        w=np.zeros((self.N,self.N))
        for k in range(len(self.train_dataset.edge_index[0])):
            w[self.train_dataset.edge_index[0][k],self.train_dataset.edge_index[1][k]] = 1

        self.m = robjects.r.matrix(FloatVector(w), nrow = self.N, ncol = self.N)
        _vts = robjects.r.matrix(
            rpyn.numpy2rpy(np.array(self.train_dataset.features).reshape(-1,1).squeeze()), 
            nrow = np.array(self.train_dataset.targets).shape[0] + self.lags, 
            ncol = self.N
        )
        self.fit = GNAR.GNARfit(vts=_vts,net = GNAR.matrixtoGNAR(self.m), alphaOrder = self.lags, betaOrder = FloatVector([1]*self.lags))
        
        self.nof_filters = None
        self.epochs = None
    def __call__(self,dataset,mode='fit',n_ahead=1):
        r_code = '''
        substitute<-function(lrnr_fit1,lrnr_fit2){
        lrnr_fit1$mod$coef = lrnr_fit2$mod$coef
        return(lrnr_fit1)
        }
        '''
        robjects.r(r_code)
        substitute=robjects.globalenv['substitute']
        _vts = robjects.r.matrix(
            rpyn.numpy2rpy(np.array(dataset.features).reshape(-1,1).squeeze()), 
            nrow = np.array(dataset.targets).shape[0] + self.lags, 
            ncol = self.N
        )
        self._fit = GNAR.GNARfit(vts = _vts, net = GNAR.matrixtoGNAR(self.m), alphaOrder = self.lags, betaOrder = FloatVector([1]*self.lags))
        self._fit = substitute(self._fit,self.fit)
        
        X = torch.tensor(dataset.features).float()
        y = torch.tensor(dataset.targets).float()
        if mode == 'fit':
            X = np.array(dataset.features)
            yhat = GNAR.fitted_GNARfit(self._fit,robjects.FloatVector(X))
            X = torch.tensor(X).float()
            yhat = torch.tensor(np.array(yhat)).float()
        elif mode == 'fore': 
            yhat = GNAR.predict_GNARfit(self.fit,n_ahead=n_ahead)
            yhat = torch.tensor(np.array(yhat)).float()
        else: 
            print('mode should be "fit" or "fore"')
        return {'X':X, 'y':y, 'yhat':yhat} 