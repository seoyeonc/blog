import numpy as np
import pandas as pd

import torch 
import torch_geometric_temporal

from .utils import convert_train_dataset


from .utils import convert_train_dataset
from .utils import ChickenpoxDatasetLoader
from .utils import PedalMeDatasetLoader
from .utils import WindmillOutputLargeDatasetLoader
from .utils import WindmillOutputMediumDatasetLoader
from .utils import WindmillOutputSmallDatasetLoader
from .utils import MontevideoBusDatasetLoader
from .utils import WikiMathsDatasetLoader
from .utils import METRLADatasetLoader
from .utils import PemsBayDatasetLoader
from .utils import EnglandCovidDatasetLoader
from .utils import MTMDatasetLoader
from .utils import TwitterTennisDatasetLoader


def padding(DL,train_dataset_miss,*args,interpolation_method='linear',num_timesteps_in=None, num_timesteps_out=None,frames=None,**kwargs):
    mindex = train_dataset_miss.mindex 
    f,lags = convert_train_dataset(train_dataset_miss)
    T,N = f.shape
    FX = pd.DataFrame(f).interpolate(method=interpolation_method,axis=0,*args,**kwargs).fillna(method='bfill').fillna(method='ffill').to_numpy().tolist()
    
    if isinstance(DL, (ChickenpoxDatasetLoader, PedalMeDatasetLoader, WindmillOutputLargeDatasetLoader, WindmillOutputMediumDatasetLoader, WindmillOutputSmallDatasetLoader, MontevideoBusDatasetLoader, WikiMathsDatasetLoader)):
        data_dict = {
        'edges':train_dataset_miss.edge_index.T.tolist(), 
        'node_ids':{'node'+str(i):i for i in range(N)}, 
        'FX':FX}
        train_dataset_padded = DL(data_dict).get_dataset(lags=lags)
        
    elif isinstance(DL, (EnglandCovidDatasetLoader)):
        data_dict = {
            'edges':[train_dataset_miss.edge_indices[j].T.tolist() for j in range(T-1)],
            'node_ids':{'node'+str(i):i for i in range(N)}, 
            'FX':FX}
        train_dataset_padded = DL(data_dict).get_dataset(lags=lags)
        
    elif isinstance(DL, (METRLADatasetLoader,PemsBayDatasetLoader)): 
        data_dict = {
            'edges':train_dataset_miss.edge_index.T.tolist(), 
            'node_ids':{'node'+str(i):i for i in range(N)}, 
            'FX':FX}
        train_dataset_padded = DL(data_dict).get_dataset(num_timesteps_in=num_timesteps_in, num_timesteps_out=num_timesteps_out)
        
    elif isinstance(DL, (MTMDatasetLoader)):
        data_dict = {
            'edges':train_dataset_miss.edge_index.T.tolist(), 
            'node_ids':{'node'+str(i):i for i in range(N)}, 
            'FX':FX}
        train_dataset_padded = DL(data_dict).get_dataset(frames=frames)
        
        
    # elif isinstance(DL, (TwitterTennisDatasetLoader)):
    #     data_dict = {
    #         'edges':train_dataset_miss.edge_index.T.tolist(), 
    #         'node_ids':{'node'+str(i):i for i in range(N)}, 
    #         'FX':FX}
    
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

def miss(DL,train_dataset,mindex,mtype,num_timesteps_in=None, num_timesteps_out=None,frames=None):
    f,lags = convert_train_dataset(train_dataset)
    T,N = f.shape
    for i,m in enumerate(mindex): 
        f[m,i] = np.nan
    if isinstance(DL, (ChickenpoxDatasetLoader, PedalMeDatasetLoader, WindmillOutputLargeDatasetLoader, WindmillOutputMediumDatasetLoader, WindmillOutputSmallDatasetLoader, MontevideoBusDatasetLoader, WikiMathsDatasetLoader)):
        data_dict = {
        'edges':train_dataset.edge_index.T.tolist(), 
        'node_ids':{'node'+str(i):i for i in range(N)}, 
        'FX':f.tolist()
    }
        train_dataset = ChickenpoxDatasetLoader(data_dict).get_dataset(lags=lags)
    elif isinstance(DL, (EnglandCovidDatasetLoader)):
        data_dict = {
            'edges':[train_dataset.edge_indices[j].T.tolist() for j in range(T-1)],
            'node_ids':{'node'+str(i):i for i in range(N)}, 
            'FX':f.tolist()
        }
        train_dataset = EnglandCovidDatasetLoader(data_dict).get_dataset(lags=lags)
    elif isinstance(DL, (METRLADatasetLoader,PemsBayDatasetLoader)): 
        data_dict = {
            'edges':train_dataset.edge_index.T.tolist(), 
            'node_ids':{'node'+str(i):i for i in range(N)}, 
            'FX':f.tolist()
        }
        train_dataset = METRLADatasetLoader(data_dict).get_dataset(num_timesteps_in=num_timesteps_in, num_timesteps_out=num_timesteps_out)
    elif isinstance(DL, (MTMDatasetLoader)):
        data_dict = {
            'edges':train_dataset.edge_index.T.tolist(), 
            'node_ids':{'node'+str(i):i for i in range(N)}, 
            'FX':f.tolist()
        }
        train_dataset = MTMDatasetLoader(data_dict).get_dataset(frames=frames)
    # elif DL == TwitterTennisDatasetLoader:
    #     data_dict = {
    #         'edges':train_dataset.edge_index.T.tolist(), 
    #         'node_ids':{'node'+str(i):i for i in range(N)}, 
    #         'FX':f
    #     }
    train_dataset.mindex = mindex
    train_dataset.mrate_eachnode = [len(mx)/T for mx in mindex]
    train_dataset.mrate_total= float(np.sum([len(mx) for mx in train_dataset.mindex])/(N*T))
    train_dataset.mtype= mtype
    return train_dataset