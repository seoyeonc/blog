from .iter import *

# torch
import torch
import torch.nn.functional as F

# return h
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.nn.recurrent import EvolveGCNH
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from torch_geometric_temporal.nn.recurrent import GCLSTM
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from torch_geometric_temporal.nn.recurrent import LRGCN
from torch_geometric_temporal.nn.recurrent import MPNNLSTM
from torch_geometric_temporal.nn.recurrent import TGCN

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
    )
    ex) lags=2, filter=32, K=2
    GConvGRU_RecurrentGCN(
      (recurrent): GConvGRU(
        (conv_x_z): ChebConv(2, 32, K=2, normalization=sym)
        (conv_h_z): ChebConv(32, 32, K=2, normalization=sym)
        (conv_x_r): ChebConv(2, 32, K=2, normalization=sym)
        (conv_h_r): ChebConv(32, 32, K=2, normalization=sym)
        (conv_x_h): ChebConv(2, 32, K=2, normalization=sym)
        (conv_h_h): ChebConv(32, 32, K=2, normalization=sym)
      )
      (linear): Linear(in_features=32, out_features=1, bias=True)
    )
    """
    def __init__(self, train_dataset, filters):
        super(GConvGRU_RecurrentGCN, self).__init__()
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.filters = filters
        self.recurrent = GConvGRU(in_channels = self.lags, out_channels = self.filters, K = 2)
        self.linear = torch.nn.Linear(self.filters, 1)
        self.iter = standard_iter()
        self.num_model = self.iter.num_model

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


        
class DCRNN_RecurrentGCN(torch.nn.Module):
    """
    Init signature: 
        DCRNN(
        in_channels: int, 
        out_channels: int, 
        K: int, 
        bias: bool = True)
    ex) lags = 4, filters = 32, K = 2
    DCRNN_RecurrentGCN(
      (recurrent): DCRNN(
        (conv_x_z): DConv(36, 32)
        (conv_x_r): DConv(36, 32)
        (conv_x_h): DConv(36, 32)
      )
      (linear): Linear(in_features=32, out_features=1, bias=True)
    )
    """
    def __init__(self, train_dataset, filters):
        super(DCRNN_RecurrentGCN, self).__init__()        
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.filters = filters
        self.recurrent = DCRNN(in_channels = self.lags, out_channels = self.filters, K = 2)
        self.linear = torch.nn.Linear(self.filters, 1)
        self.iter = accumulated_iter_first()
        self.num_model = self.iter.num_model

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

class EvolveGCNH_RecurrentGCN(torch.nn.Module):
    """
    Init signature:
        EvolveGCNH_RecurrentGCN(
            num_of_nodes: int,
            in_channels: int,
            improved: bool = False,
            cached: bool = False,
            normalize: bool = True,
            add_self_loops: bool = True,
        )
    ex) lags = 4, num_of_nodes = 20
    RecurrentGCN(
      (recurrent): EvolveGCNH(
        (pooling_layer): TopKPooling(4, ratio=0.2, multiplier=1.0)
        (recurrent_layer): GRU(4, 4)
        (conv_layer): GCNConv_Fixed_W(4, 4)
      )
      (linear): Linear(in_features=4, out_features=1, bias=True)
    )
    """
    def __init__(self, train_dataset, filters = None):
        super(EvolveGCNH_RecurrentGCN, self).__init__()            
        f, self.lags = convert_train_dataset(train_dataset)
        _, self.node_count = f.shape
        self.recurrent = EvolveGCNH(num_of_nodes = self.node_count, in_channels = self.lags)
        self.linear = torch.nn.Linear(self.lags, 1)
        self.filters = filters
        self.iter = accumulated_iter_first()
        self.num_model = self.iter.num_model

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
        
class EvolveGCNO_RecurrentGCN(torch.nn.Module):
    """Init signature:
        EvolveGCNO(
            in_channels: int,
            improved: bool = False,
            cached: bool = False,
            normalize: bool = True,
            add_self_loops: bool = True,
        )
    ex) lags = 4
    EvolveGCNO_RecurrentGCN(
      (recurrent): EvolveGCNO(
        (recurrent_layer): GRU(4, 4)
        (conv_layer): GCNConv_Fixed_W(4, 4)
      )
      (linear): Linear(in_features=4, out_features=1, bias=True)
    )
        """
    def __init__(self, train_dataset, filters = None):
        super(EvolveGCNO_RecurrentGCN, self).__init__()
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.recurrent = EvolveGCNO(in_channels = self.lags)
        self.linear = torch.nn.Linear(self.lags, 1)
        self.filters = filters
        self.iter = accumulated_iter_first_grad()
        self.num_model = self.iter.num_model

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
        
class GCLSTM_RecurrentGCN(torch.nn.Module):
    """
    Init signature:
        GCLSTM(
            in_channels: int,
            out_channels: int,
            K: int,
            normalization: str = 'sym',
            bias: bool = True,
        )
    ex) lags = 4, filters = 32, K = 1
    GCLSTM_RecurrentGCN(
      (recurrent): GCLSTM(
        (conv_i): ChebConv(32, 32, K=1, normalization=sym)
        (conv_f): ChebConv(32, 32, K=1, normalization=sym)
        (conv_c): ChebConv(32, 32, K=1, normalization=sym)
        (conv_o): ChebConv(32, 32, K=1, normalization=sym)
      )
      (linear): Linear(in_features=32, out_features=1, bias=True)
    )
    """
    def __init__(self, train_dataset, filters):
        super(GCLSTM_RecurrentGCN, self).__init__()
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.filters = filters
        self.recurrent = GCLSTM(in_channels = self.lags, out_channels = self.filters, K = 1)
        self.linear = torch.nn.Linear(self.filters, 1)
        self.iter = accumulated_iter_second()
        self.num_model = self.iter.num_model

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0
        
        
class GConvLSTM_RecurrentGCN(torch.nn.Module):
    """
    Init signature:
    GConvLSTM(
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = 'sym',
        bias: bool = True,
    ) lags = 4, filters = 32, K = 1
    ex)
    GConvLSTM_RecurrentGCN(
      (recurrent): GConvLSTM(
        (conv_x_i): ChebConv(4, 32, K=1, normalization=sym)
        (conv_h_i): ChebConv(32, 32, K=1, normalization=sym)
        (conv_x_f): ChebConv(4, 32, K=1, normalization=sym)
        (conv_h_f): ChebConv(32, 32, K=1, normalization=sym)
        (conv_x_c): ChebConv(4, 32, K=1, normalization=sym)
        (conv_h_c): ChebConv(32, 32, K=1, normalization=sym)
        (conv_x_o): ChebConv(4, 32, K=1, normalization=sym)
        (conv_h_o): ChebConv(32, 32, K=1, normalization=sym)
      )
      (linear): Linear(in_features=32, out_features=1, bias=True)
    )
    """
    def __init__(self, train_dataset, filters):
        super(GConvLSTM_RecurrentGCN, self).__init__()
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.filters = filters
        self.recurrent = GConvLSTM(in_channels = self.lags, out_channels = self.filters, K = 1)
        self.linear = torch.nn.Linear(self.filters, 1)
        self.iter = accumulated_iter_second()
        self.num_model = self.iter.num_model

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0
            
class LRGCN_RecurrentGCN(torch.nn.Module):
    """
    Init signature:
    LRGCN(
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int,
    )
    ex) lags = 4, filters = 32, num_relations = 1, num_bases = 1
    LRGCN_RecurrentGCN(
      (recurrent): LRGCN(
        (conv_x_i): RGCNConv(4, 32, num_relations=1)
        (conv_h_i): RGCNConv(32, 32, num_relations=1)
        (conv_x_f): RGCNConv(4, 32, num_relations=1)
        (conv_h_f): RGCNConv(32, 32, num_relations=1)
        (conv_x_c): RGCNConv(4, 32, num_relations=1)
        (conv_h_c): RGCNConv(32, 32, num_relations=1)
        (conv_x_o): RGCNConv(4, 32, num_relations=1)
        (conv_h_o): RGCNConv(32, 32, num_relations=1)
      )
      (linear): Linear(in_features=32, out_features=1, bias=True)
    )
    """
    def __init__(self, train_dataset, filters):
        super(LRGCN_RecurrentGCN, self).__init__()
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.filters = filters
        self.recurrent = LRGCN(in_channels = self.lags, out_channels = self.filters, num_relations = 1, num_bases = 1)
        self.linear = torch.nn.Linear(self.filters, 1)
        self.iter = accumulated_iter_second()
        self.num_model = self.iter.num_model

    def forward(self, x, edge_index, edge_weight, h_0, c_0):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h_0, c_0)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0
    
class MPNNLSTM_RecurrentGCN(torch.nn.Module):
    """
    Init signature:
        MPNNLSTM(
        in_channels: int,
        hidden_size: int,
        num_nodes: int,
        window: int,
        dropout: float,
    )
    RecurrentGCN(
      (recurrent): MPNNLSTM(
        (_convolution_1): GCNConv(lags, 32)
        (_convolution_2): GCNConv(32, 32)
        (_batch_norm_1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (_batch_norm_2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (_recurrent_1): LSTM(64, 32)
        (_recurrent_2): LSTM(32, 32)
      )
      (linear): Linear(in_features=32*2+lags, out_features=1, bias=True)
    )
    """
    def __init__(self, train_dataset, filters):
        super(MPNNLSTM_RecurrentGCN, self).__init__()
        f, self.lags = convert_train_dataset(train_dataset)
        _, self.node_count = f.shape
        self.filters = filters
        self.recurrent = MPNNLSTM(in_channels = self.lags, hidden_size = 32, num_nodes = self.node_count, window = 1, dropout = 0.5)
        self.linear = torch.nn.Linear(2*32 + self.lags, 1)
        self.iter = accumulated_iter_first()
        self.num_model = self.iter.num_model
        

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

class TGCN_RecurrentGCN(torch.nn.Module):
    """
    Init signature:
    TGCN(
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    )
    ex) lags = 4, 
    TGCN_RecurrentGCN(
      (recurrent): TGCN(
        (conv_z): GCNConv(4, 32)
        (linear_z): Linear(in_features=64, out_features=32, bias=True)
        (conv_r): GCNConv(4, 32)
        (linear_r): Linear(in_features=64, out_features=32, bias=True)
        (conv_h): GCNConv(4, 32)
        (linear_h): Linear(in_features=64, out_features=32, bias=True)
      )
      (linear): Linear(in_features=32, out_features=1, bias=True)
    )
    """
    def __init__(self, train_dataset, filters):
        super(TGCN_RecurrentGCN, self).__init__()
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.filters = filters
        self.recurrent = TGCN(in_channels = self.lags, out_channels = self.filters)
        self.linear = torch.nn.Linear(self.filters, 1)
        self.iter = accumulated_iter_third()
        self.num_model = self.iter.num_model

    def forward(self, x, edge_index, edge_weight, prev_hidden_state):
        h = self.recurrent(x, edge_index, edge_weight, prev_hidden_state)
        y = F.relu(h)
        y = self.linear(y)
        return y, h
        
