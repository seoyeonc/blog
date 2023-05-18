import numpy as np
import pandas as pd

import torch 
import torch_geometric_temporal

from .utils import DatasetLoader
from .utils import convert_train_dataset


def padding(train_dataset_miss,*args,interpolation_method='linear',**kwargs):
    mindex = train_dataset_miss.mindex 
    f,lags = convert_train_dataset(train_dataset_miss)
    T,N = f.shape
    FX = pd.DataFrame(f).interpolate(method=interpolation_method,axis=0,*args,**kwargs).fillna(method='bfill').fillna(method='ffill').to_numpy().tolist()
    data_dict = {
        'edges':train_dataset_miss.edge_index.T.tolist(), 
        'node_ids':{'node'+str(i):i for i in range(N)}, 
        'FX':FX
    }
    train_dataset_padded = DatasetLoader(data_dict).get_dataset(lags=lags)
    train_dataset_padded.mindex = mindex
    train_dataset_padded.mrate_eachnode = train_dataset_miss.mrate_eachnode
    train_dataset_padded.mrate_total = train_dataset_miss.mrate_total
    train_dataset_padded.mtype= train_dataset_miss.mtype
    train_dataset_padded.interpolation_method = interpolation_method
    return train_dataset_padded

def rand_mindex(train_dataset,mrate = 0.5):
    f,lags = convert_train_dataset(train_dataset)
    T,N = f.shape
    missing_count = int(np.round(mrate*T,0))
    mindex = [np.sort(np.random.choice(range(T),missing_count,replace=False)).tolist() for i in range(N)]  
    return mindex

def miss(train_dataset,mindex,mtype):
    f,lags = convert_train_dataset(train_dataset)
    T,N = f.shape
    for i,m in enumerate(mindex): 
        f[m,i] = np.nan
    data_dict = {
        'edges':train_dataset.edge_index.T.tolist(), 
        'node_ids':{'node'+str(i):i for i in range(N)}, 
        'FX':f.tolist()
    }
    train_dataset = DatasetLoader(data_dict).get_dataset(lags=lags)
    train_dataset.mindex = mindex
    train_dataset.mrate_eachnode = [len(mx)/T for mx in mindex]
    train_dataset.mrate_total= float(np.sum([len(mx) for mx in train_dataset.mindex])/(N*T))
    train_dataset.mtype= mtype
    return train_dataset