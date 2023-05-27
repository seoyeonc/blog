# torch
import torch
import torch.nn.functional as F
# return h
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.nn.recurrent import EvolveGCNH
from torch_geometric_temporal.nn.recurrent import EvolveGCNO

from .utils import convert_train_dataset

class GConvGRU_RecurrentGCN(torch.nn.Module):
    """
    Init signature:
        GConvGRU(
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = 'sym',
        bias: bool = True,
    )"""
    def __init__(self, train_dataset, filters):
        super(GConvGRU_RecurrentGCN, self).__init__()
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.filters = filters
        self.recurrent = GConvGRU(in_channels = self.lags, out_channels = self.filters, K = 2)
        self.linear = torch.nn.Linear(self.filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

class GConvGRU_iter:
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        
    def do_iter(self,model,optimizer):
        for t, snapshot in enumerate(self.train_dataset):
            yt_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).reshape(-1)
            cost = torch.mean((yt_hat-snapshot.y)**2)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
        
class DCRNN_RecurrentGCN(torch.nn.Module):
    """
    Init signature: 
        DCRNN(
        in_channels: int, 
        out_channels: int, 
        K: int, 
        bias: bool = True)
    """
    def __init__(self, train_dataset, filters):
        super(DCRNN_RecurrentGCN, self).__init__()        
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.filters = filters
        self.recurrent = DCRNN(in_channels = self.lags, out_channels = self.filters, K = 2)
        self.linear = torch.nn.Linear(self.filters, 1)        

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
class DCRNN_iter:
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        
    def do_iter(self, model, optimizer):
        cost = 0
        for t, snapshot in enumerate(self.train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).reshape(-1)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (t+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
    

class EvolveGCNH_RecurrentGCN(torch.nn.Module):
    """Init signature:
        EvolveGCNH(
            num_of_nodes: int,
            in_channels: int,
            improved: bool = False,
            cached: bool = False,
            normalize: bool = True,
            add_self_loops: bool = True,
        )
    """
    def __init__(self, train_dataset, filters = None):
        super(EvolveGCNH_RecurrentGCN, self).__init__()            
        f, self.lags = convert_train_dataset(train_dataset)
        _, self.node_count = f.shape
        self.recurrent = EvolveGCNH(num_of_nodes = self.node_count, in_channels = self.lags)
        self.linear = torch.nn.Linear(self.lags, 1)
        self.filters = filters

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
class EvolveGCNH_iter:
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        
    def do_iter(self, model, optimizer): 
        cost = 0
        for time, snapshot in enumerate(self.train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).reshape(-1)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (time+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        
class EvolveGCNO_RecurrentGCN(torch.nn.Module):
    """Init signature:
        EvolveGCNO(
            in_channels: int,
            improved: bool = False,
            cached: bool = False,
            normalize: bool = True,
            add_self_loops: bool = True,
        )
        """
    def __init__(self, train_dataset, filters = None):
        super(EvolveGCNO_RecurrentGCN, self).__init__()
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.recurrent = EvolveGCNO(in_channels = self.lags)
        self.linear = torch.nn.Linear(self.lags, 1)
        self.filters = filters

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
class EvolveGCNO_iter:
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        
    def do_iter(self, model, optimizer): 
        cost = 0
        for time, snapshot in enumerate(self.train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).reshape(-1)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (time+1)
        cost.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()