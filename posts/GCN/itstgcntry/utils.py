import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib import animation

# # torch
import torch
# import torch.nn.functional as F
import torch_geometric_temporal
# from torch_geometric_temporal.nn.recurrent import GConvGRU

# # scipy 
# from scipy.interpolate import interp1d

# utils
#import copy
#import time
import pickle
import itertools
#from tqdm import tqdm
#import warnings

# rpy2
#import rpy2
#import rpy2.robjects as ro 
#from rpy2.robjects.vectors import FloatVector
#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr
#import rpy2.robjects.numpy2ri as rpyn
#GNAR = importr('GNAR') # import GNAR 
#igraph = importr('igraph') # import igraph 
#ebayesthresh = importr('EbayesThresh').ebayesthresh

import os
from typing import List

temporal_signal_split = torch_geometric_temporal.signal.temporal_signal_split


def load_data(fname):
    with open(fname, 'rb') as outfile:
        data_dict = pickle.load(outfile)
    return data_dict

def save_data(data_dict,fname):
    with open(fname,'wb') as outfile:
        pickle.dump(data_dict,outfile)
        

def plot(f,*args,t=None,h=2.5,**kwargs):
    T,N = f.shape
    if t is None: t = range(T)
    fig = plt.figure()
    ax = fig.subplots(N,1)
    for n in range(N):
        ax[n].plot(t,f[:,n],*args,**kwargs)
        ax[n].set_title('node='+str(n))
    fig.set_figheight(N*h)
    fig.tight_layout()
    plt.close()
    return fig

def plot_add(fig,f,*args,t=None,**kwargs):
    T = f.shape[0]
    N = f.shape[1] 
    if t is None: t = range(T)   
    ax = fig.get_axes()
    for n in range(N):
        ax[n].plot(t,f[:,n],*args,**kwargs)
    return fig

def convert_train_dataset(train_dataset,Datatyp):
    lags = torch.tensor(train_dataset.features).shape[-1]
    if Datatyp=='StaticGraphTemporalSignal_lags':
        f = torch.concat([train_dataset[0].x.T,torch.tensor(train_dataset.targets)],axis=0).numpy()

    elif Datatyp=='StaticGraphTemporalSignal_timestamps':
        f = torch.concat([train_dataset[0].x.T.reshape(-1,torch.tensor(train_dataset.targets).shape[1]),torch.tensor(train_dataset.targets).reshape(-1,torch.tensor(train_dataset.targets).shape[1])],axis=0).numpy()

    elif Datatyp=='DynamicGraphTemporalSignal':
        f = torch.concat([train_dataset[0].x.T,torch.tensor(train_dataset.targets)],axis=0).numpy()
        
    else:
        f = torch.concat([train_dataset[0].x.T,torch.tensor(train_dataset.targets)],axis=0).numpy()
    
    return f,lags 

class ChickenpoxDatasetLoader(object):
    def __init__(self,data_dict):
        self._dataset = data_dict 
    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["FX"])
        self.features = [
            stacked_target[i : i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]
        self.targets = [
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 4) -> torch_geometric_temporal.signal.StaticGraphTemporalSignal:
        """Returning the Chickenpox Hungary data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(torch_geometric_temporal.signal.StaticGraphTemporalSignal)* - The Chickenpox Hungary dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = torch_geometric_temporal.signal.StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset


class EnglandCovidDatasetLoader(object):
    """A dataset of mobility and history of reported cases of COVID-19
    in England NUTS3 regions, from 3 March to 12 of May. The dataset is
    segmented in days and the graph is directed and weighted. The graph
    indicates how many people moved from one region to the other each day,
    based on Facebook Data For Good disease prevention maps.
    The node features correspond to the number of COVID-19 cases
    in the region in the past **window** days. The task is to predict the
    number of cases in each node after 1 day. For details see this paper:
    `"Transfer Graph Neural Networks for Pandemic Forecasting." <https://arxiv.org/abs/2009.08388>`_
    """

    def __init__(self):
        self._read_web_data()

    def _get_edges(self):
        self._edges = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._edges.append(
                np.array(self._dataset["edge_mapping"]["edge_index"][str(time)]).T
            )

    def _get_edge_weights(self):
        self._edge_weights = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._edge_weights.append(
                np.array(self._dataset["edge_mapping"]["edge_weight"][str(time)])
            )

    def _get_targets_and_features(self):

        stacked_target = np.array(self._dataset["y"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10 ** -10
        )
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]

    def get_dataset(self, lags: int = 8) -> torch_geometric_temporal.signal.dynamic_graph_temporal_signal.DynamicGraphTemporalSignal:
        """Returning the England COVID19 data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The England Covid dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = torch_geometric_temporal.signal.dynamic_graph_temporal_signal.DynamicGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
    
class METRLADatasetLoader(object):
    """A traffic forecasting dataset based on Los Angeles
    Metropolitan traffic conditions. The dataset contains traffic
    readings collected from 207 loop detectors on highways in Los Angeles
    County in aggregated 5 minute intervals for 4 months between March 2012
    to June 2012.

    For further details on the version of the sensor network and
    discretization see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_
    """

    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data")):
        super(METRLADatasetLoader, self).__init__()
        self.raw_data_dir = raw_data_dir
        self._read_web_data()

    def _download_url(self, url, save_path):  # pragma: no cover
        with urllib.request.urlopen(url) as dl_file:
            with open(save_path, "wb") as out_file:
                out_file.write(dl_file.read())

    def _read_web_data(self):
        url = "https://graphmining.ai/temporal_datasets/METR-LA.zip"

        # Check if zip file is in data folder from working directory, otherwise download
        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "METR-LA.zip")
        ):  # pragma: no cover
            if not os.path.exists(self.raw_data_dir):
                os.makedirs(self.raw_data_dir)
            self._download_url(url, os.path.join(self.raw_data_dir, "METR-LA.zip"))

        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "adj_mat.npy")
        ) or not os.path.isfile(
            os.path.join(self.raw_data_dir, "node_values.npy")
        ):  # pragma: no cover
            with zipfile.ZipFile(
                os.path.join(self.raw_data_dir, "METR-LA.zip"), "r"
            ) as zip_fh:
                zip_fh.extractall(self.raw_data_dir)

        A = np.load(os.path.join(self.raw_data_dir, "adj_mat.npy"))
        X = np.load(os.path.join(self.raw_data_dir, "node_values.npy")).transpose(
            (1, 2, 0)
        )
        X = X.astype(np.float32)

        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ) -> torch_geometric_temporal.signal.dynamic_graph_temporal_signal.DynamicGraphTemporalSignal:
        """Returns data iterator for METR-LA dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The METR-LA traffic
                forecasting dataset.
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = torch_geometric_temporal.signal.dynamic_graph_temporal_signal.DynamicGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset


class MontevideoBusDatasetLoader(object):
    """A dataset of inflow passenger at bus stop level from Montevideo city.
    This dataset comprises hourly inflow passenger data at bus stop level for 11 bus lines during
    October 2020 from Montevideo city (Uruguay). The bus lines selected are the ones that carry
    people to the center of the city and they load more than 25% of the total daily inflow traffic.
    Vertices are bus stops, edges are links between bus stops when a bus line connects them and the
    weight represent the road distance. The target is the passenger inflow. This is a curated
    dataset made from different data sources of the Metropolitan Transportation System (STM) of
    Montevideo. These datasets are freely available to anyone in the National Catalog of Open Data
    from the government of Uruguay (https://catalogodatos.gub.uy/).
    """

    def __init__(self):
        self._read_web_data()

    def _get_node_ids(self):
        return [node.get('bus_stop') for node in self._dataset["nodes"]]

    def _get_edges(self):
        node_ids = self._get_node_ids()
        node_id_map = dict(zip(node_ids, range(len(node_ids))))
        self._edges = np.array(
            [(node_id_map[d["source"]], node_id_map[d["target"]]) for d in self._dataset["links"]]
        ).T

    def _get_edge_weights(self):
        self._edge_weights = np.array([(d["weight"]) for d in self._dataset["links"]]).T

    def _get_features(self, feature_vars: List[str] = ["y"]):
        features = []
        for node in self._dataset["nodes"]:
            X = node.get("X")
            for feature_var in feature_vars:
                features.append(np.array(X.get(feature_var)))
        stacked_features = np.stack(features).T
        standardized_features = (
            stacked_features - np.mean(stacked_features, axis=0)
        ) / np.std(stacked_features, axis=0)
        self.features = [
            standardized_features[i : i + self.lags, :].T
            for i in range(len(standardized_features) - self.lags)
        ]

    def _get_targets(self, target_var: str = "y"):
        targets = []
        for node in self._dataset["nodes"]:
            y = node.get(target_var)
            targets.append(np.array(y))
        stacked_targets = np.stack(targets).T
        standardized_targets = (
            stacked_targets - np.mean(stacked_targets, axis=0)
        ) / np.std(stacked_targets, axis=0)
        self.targets = [
            standardized_targets[i + self.lags, :].T
            for i in range(len(standardized_targets) - self.lags)
        ]

    def get_dataset(
        self, lags: int = 4, target_var: str = "y", feature_vars: List[str] = ["y"]) -> torch_geometric_temporal.signal.StaticGraphTemporalSignal:
        """Returning the MontevideoBus passenger inflow data iterator.

        Parameters
        ----------
        lags : int, optional
            The number of time lags, by default 4.
        target_var : str, optional
            Target variable name, by default "y".
        feature_vars : List[str], optional
            List of feature variables, by default ["y"].

        Returns
        -------
        StaticGraphTemporalSignal
            The MontevideoBus dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_features(feature_vars)
        self._get_targets(target_var)
        dataset = torch_geometric_temporal.signal.StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset


class MTMDatasetLoader:
    """
    A dataset of `Methods-Time Measurement-1 <https://en.wikipedia.org/wiki/Methods-time_measurement>`_
    (MTM-1) motions, signalled as consecutive video frames of 21 3D hand keypoints, acquired via
    `MediaPipe Hands <https://google.github.io/mediapipe/solutions/hands.html>`_ from RGB-Video
    material. Vertices are the finger joints of the human hand and edges are the bones connecting
    them. The targets are manually labeled for each frame, according to one of the five MTM-1
    motions (classes :math:`C`): Grasp, Release, Move, Reach, Position plus a negative class for
    frames without graph signals (no hand present). This is a classification task where :math:`T`
    consecutive frames need to be assigned to the corresponding class :math:`C`. The data x is
    returned in shape :obj:`(3, 21, T)`, the target is returned one-hot-encoded in shape :obj:`(T, 6)`.
    """

    def __init__(self):
        self._read_web_data()

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array([1 for d in self._dataset["edges"]]).T

    def _get_features(self):
        dic = self._dataset
        joints = [str(n) for n in range(21)]
        dataset_length = len(dic["0"].values())
        features = np.zeros((dataset_length, 21, 3))

        for j, joint in enumerate(joints):
            for t, xyz in enumerate(dic[joint].values()):
                xyz_tuple = list(map(float, xyz.strip("()").split(",")))
                features[t, j, :] = xyz_tuple

        self.features = [
            features[i : i + self.frames, :].T
            for i in range(len(features) - self.frames)
        ]

    def _get_targets(self):
        # target eoncoding: {0 : 'Grasp', 1 : 'Move', 2 : 'Negative',
        #                   3 : 'Position', 4 : 'Reach', 5 : 'Release'}
        targets = []
        for _, y in self._dataset["LABEL"].items():
            targets.append(y)

        n_values = np.max(targets) + 1
        targets_ohe = np.eye(n_values)[targets]

        self.targets = [
            targets_ohe[i : i + self.frames, :]
            for i in range(len(targets_ohe) - self.frames)
        ]

    def get_dataset(self, frames: int = 16) -> torch_geometric_temporal.signal.StaticGraphTemporalSignal:
        """Returning the MTM-1 motion data iterator.

        Args types:
            * **frames** *(int)* - The number of consecutive frames T, default 16.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The MTM-1 dataset.
        """
        self.frames = frames
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()

        dataset = torch_geometric_temporal.signal.StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset

class PedalMeDatasetLoader(object):
    """A dataset of PedalMe Bicycle deliver orders in London between 2020
    and 2021. We made it public during the development of PyTorch Geometric
    Temporal. The underlying graph is static - vertices are localities and
    edges are spatial_connections. Vertex features are lagged weekly counts of the
    delivery demands (we included 4 lags). The target is the weekly number of
    deliveries the upcoming week. Our dataset consist of more than 30 snapshots (weeks).
    """

    def __init__(self):
        self._read_web_data()

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["X"])
        self.features = [
            stacked_target[i : i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]
        self.targets = [
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 4) -> torch_geometric_temporal.signal.StaticGraphTemporalSignal:
        """Returning the PedalMe London demand data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The PedalMe dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = torch_geometric_temporal.signal.StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
    
class PemsBayDatasetLoader(object):
    """A traffic forecasting dataset as described in Diffusion Convolution Layer Paper.

    This traffic dataset is collected by California Transportation Agencies (CalTrans)
    Performance Measurement System (PeMS). It is represented by a network of 325 traffic sensors
    in the Bay Area with 6 months of traffic readings ranging from Jan 1st 2017 to May 31th 2017
    in 5 minute intervals.

    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_
    """

    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data")):
        super(PemsBayDatasetLoader, self).__init__()
        self.raw_data_dir = raw_data_dir
        self._read_web_data()

    def _download_url(self, url, save_path):  # pragma: no cover
        with urllib.request.urlopen(url) as dl_file:
            with open(save_path, "wb") as out_file:
                out_file.write(dl_file.read())

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, :, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12) -> torch_geometric_temporal.signal.StaticGraphTemporalSignal:
        """Returns data iterator for PEMS-BAY dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The PEMS-BAY traffic
                forecasting dataset.
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = torch_geometric_temporal.signal.StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )
        return dataset


def transform_degree(x, cutoff=4):
    log_deg = np.ceil(np.log(x + 1.0))
    return np.minimum(log_deg, cutoff)


def transform_transitivity(x):
    trans = x * 10
    return np.floor(trans)


def onehot_encoding(x, unique_vals):
    E = np.zeros((len(x), len(unique_vals)))
    for i, val in enumerate(x):
        E[i, unique_vals.index(val)] = 1.0
    return E


def encode_features(X, log_degree_cutoff=4):
    X_arr = np.array(X)
    a = transform_degree(X_arr[:, 0], log_degree_cutoff)
    b = transform_transitivity(X_arr[:, 1])
    A = onehot_encoding(a, range(log_degree_cutoff + 1))
    B = onehot_encoding(b, range(11))
    return np.concatenate((A, B), axis=1)


class TwitterTennisDatasetLoader(object):
    """
    Twitter mention graphs related to major tennis tournaments from 2017.
    Nodes are Twitter accounts and edges are mentions between them.
    Each snapshot contains the graph induced by the most popular nodes
    of the original dataset. Node labels encode the number of mentions
    received in the original dataset for the next snapshot. Read more
    on the original Twitter data in the 'Temporal Walk Based Centrality Metric for Graph Streams' paper.

    Parameters
    ----------
    event_id : str
        Choose to load the mention network for Roland-Garros 2017 ("rg17") or USOpen 2017 ("uo17")
    N : int <= 1000
        Number of most popular nodes to load. By default N=1000. Each snapshot contains the graph induced by these nodes.
    feature_mode : str
        None : load raw degree and transitivity node features
        "encoded" : load onehot encoded degree and transitivity node features
        "diagonal" : set identity matrix as node features
    target_offset : int
        Set the snapshot offset for the node labels to be predicted. By default node labels for the next snapshot are predicted (target_offset=1).
    """

    def __init__(
        self, event_id="rg17", N=None, feature_mode="encoded", target_offset=1
    ):
        self.N = N
        self.target_offset = target_offset
        if event_id in ["rg17", "uo17"]:
            self.event_id = event_id
        else:
            raise ValueError(
                "Invalid 'event_id'! Choose 'rg17' or 'uo17' to load the Roland-Garros 2017 or the USOpen 2017 Twitter tennis dataset respectively."
            )
        if feature_mode in [None, "diagonal", "encoded"]:
            self.feature_mode = feature_mode
        else:
            raise ValueError(
                "Choose feature_mode from values [None, 'diagonal', 'encoded']."
            )
        self._read_web_data()

    def _get_edges(self):
        edge_indices = []
        self.edges = []
        for time in range(self._dataset["time_periods"]):
            E = np.array(self._dataset[str(time)]["edges"])
            if self.N != None:
                selector = np.where((E[:, 0] < self.N) & (E[:, 1] < self.N))
                E = E[selector]
                edge_indices.append(selector)
            self.edges.append(E.T)
        self.edge_indices = edge_indices

    def _get_edge_weights(self):
        edge_indices = self.edge_indices
        self.edge_weights = []
        for i, time in enumerate(range(self._dataset["time_periods"])):
            W = np.array(self._dataset[str(time)]["weights"])
            if self.N != None:
                W = W[edge_indices[i]]
            self.edge_weights.append(W)

    def _get_features(self):
        self.features = []
        for time in range(self._dataset["time_periods"]):
            X = np.array(self._dataset[str(time)]["X"])
            if self.N != None:
                X = X[: self.N]
            if self.feature_mode == "diagonal":
                X = np.identity(X.shape[0])
            elif self.feature_mode == "encoded":
                X = encode_features(X)
            self.features.append(X)

    def _get_targets(self):
        self.targets = []
        T = self._dataset["time_periods"]
        for time in range(T):
            # predict node degrees in advance
            snapshot_id = min(time + self.target_offset, T - 1)
            y = np.array(self._dataset[str(snapshot_id)]["y"])
            # logarithmic transformation for node degrees
            y = np.log(1.0 + y)
            if self.N != None:
                y = y[: self.N]
            self.targets.append(y)

    def get_dataset(self) -> torch_geometric_temporal.signal.dynamic_graph_temporal_signal.DynamicGraphTemporalSignal:
        """Returning the TennisDataset data iterator.

        Return types:
            * **dataset** *(DynamicGraphTemporalSignal)* - Selected Twitter tennis dataset (Roland-Garros 2017 or USOpen 2017).
        """
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()
        dataset = torch_geometric_temporal.signal.dynamic_graph_temporal_signal.DynamicGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )
        return dataset



class WikiMathsDatasetLoader(object):
    """A dataset of vital mathematics articles from Wikipedia. We made it
    public during the development of PyTorch Geometric Temporal. The
    underlying graph is static - vertices are Wikipedia pages and edges are
    links between them. The graph is directed and weighted. Weights represent
    the number of links found at the source Wikipedia page linking to the target
    Wikipedia page. The target is the daily user visits to the Wikipedia pages
    between March 16th 2019 and March 15th 2021 which results in 731 periods.
    """

    def __init__(self):
        self._read_web_data()

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):

        targets = []
        for time in range(self._dataset["time_periods"]):
            targets.append(np.array(self._dataset[str(time)]["y"]))
        stacked_target = np.stack(targets)
        standardized_target = (
            stacked_target - np.mean(stacked_target, axis=0)
        ) / np.std(stacked_target, axis=0)
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(len(targets) - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(len(targets) - self.lags)
        ]

    def get_dataset(self, lags: int = 8) -> torch_geometric_temporal.signal.StaticGraphTemporalSignal:
        """Returning the Wikipedia Vital Mathematics data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Wiki Maths dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = torch_geometric_temporal.signal.StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset

class WindmillOutputLargeDatasetLoader(object):
    """Hourly energy output of windmills from a European country
    for more than 2 years. Vertices represent 319 windmills and
    weighted edges describe the strength of relationships. The target
    variable allows for regression tasks.
    """

    def __init__(self):
        self._read_web_data()
        
    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        stacked_target = np.stack(self._dataset["block"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10 ** -10
        )
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 8) -> torch_geometric_temporal.signal.StaticGraphTemporalSignal:
        """Returning the Windmill Output data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Windmill Output dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = torch_geometric_temporal.signal.StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
    


class WindmillOutputMediumDatasetLoader(object):
    """Hourly energy output of windmills from a European country
    for more than 2 years. Vertices represent 26 windmills and
    weighted edges describe the strength of relationships. The target
    variable allows for regression tasks.
    """

    def __init__(self):
        self._read_web_data()

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        stacked_target = np.stack(self._dataset["block"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10 ** -10
        )
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 8) -> torch_geometric_temporal.signal.StaticGraphTemporalSignal:
        """Returning the Windmill Output data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Windmill Output dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = torch_geometric_temporal.signal.StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset

    
class WindmillOutputSmallDatasetLoader(object):
    """Hourly energy output of windmills from a European country
    for more than 2 years. Vertices represent 11 windmills and
    weighted edges describe the strength of relationships. The target
    variable allows for regression tasks.
    """

    def __init__(self):
        self._read_web_data()

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        stacked_target = np.stack(self._dataset["block"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10 ** -10
        )
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 8) -> torch_geometric_temporal.signal.StaticGraphTemporalSignal:
        """Returning the Windmill Output data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Windmill Output dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = torch_geometric_temporal.signal.StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
    
class Evaluator:
    def __init__(self,learner,train_dataset,test_dataset):
        self.learner = learner
        # self.learner.model.eval()
        try:self.learner.model.eval()
        except:pass
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lags = self.learner.lags
        rslt_tr = self.learner(self.train_dataset) 
        rslt_test = self.learner(self.test_dataset)
        self.X_tr = rslt_tr['X']
        self.y_tr = rslt_tr['y']
        self.f_tr = torch.concat([self.train_dataset[0].x.T,self.y_tr],axis=0).float()
        self.yhat_tr = rslt_tr['yhat']
        self.fhat_tr = torch.concat([self.train_dataset[0].x.T,self.yhat_tr],axis=0).float()
        self.X_test = rslt_test['X']
        self.y_test = rslt_test['y']
        self.f_test = self.y_test 
        self.yhat_test = rslt_test['yhat']
        self.fhat_test = self.yhat_test
        self.f = torch.concat([self.f_tr,self.f_test],axis=0)
        self.fhat = torch.concat([self.fhat_tr,self.fhat_test],axis=0)
    def calculate_mse(self):
        test_base_mse_eachnode = ((self.y_test - self.y_test.mean(axis=0).reshape(-1,self.y_test.shape[-1]))**2).mean(axis=0).tolist()
        test_base_mse_total = ((self.y_test - self.y_test.mean(axis=0).reshape(-1,self.y_test.shape[-1]))**2).mean().item()
        train_mse_eachnode = ((self.y_tr-self.yhat_tr)**2).mean(axis=0).tolist()
        train_mse_total = ((self.y_tr-self.yhat_tr)**2).mean().item()
        test_mse_eachnode = ((self.y_test-self.yhat_test)**2).mean(axis=0).tolist()
        test_mse_total = ((self.y_test-self.yhat_test)**2).mean().item()
        self.mse = {'train': {'each_node': train_mse_eachnode, 'total': train_mse_total},
                    'test': {'each_node': test_mse_eachnode, 'total': test_mse_total},
                    'test(base)': {'each_node': test_base_mse_eachnode, 'total': test_base_mse_total},
                   }
    def _plot(self,*args,t=None,h=2.5,max_node=5,**kwargs):
        T,N = self.f.shape
        if t is None: t = range(T)
        fig = plt.figure()
        nof_axs = max(min(N,max_node),2)
        if min(N,max_node)<2: 
            print('max_node should be >=2')
        ax = fig.subplots(nof_axs ,1)
        for n in range(nof_axs):
            ax[n].plot(t,self.f[:,n],color='gray',*args,**kwargs)
            ax[n].set_title('node='+str(n))
        fig.set_figheight(nof_axs*h)
        fig.tight_layout()
        plt.close()
        return fig
    def plot(self,*args,t=None,h=2.5,**kwargs):
        self.calculate_mse()
        fig = self._plot(*args,t=None,h=2.5,**kwargs)
        ax = fig.get_axes()
        for i,a in enumerate(ax):
            _mse1= self.mse['train']['each_node'][i]
            _mse2= self.mse['test']['each_node'][i]
            _mse3= self.mse['test(base)']['each_node'][i]
            _mrate = self.learner.mrate_eachnode if set(dir(self.learner.mrate_eachnode)) & {'__getitem__'} == set() else self.learner.mrate_eachnode[i]
            _title = 'node{0}, mrate: {1:.2f}% \n mse(train) = {2:.2f}, mse(test) = {3:.2f}, mse(test_base) = {4:.2f}'.format(i,_mrate*100,_mse1,_mse2,_mse3)
            a.set_title(_title)
            _t1 = self.lags
            _t2 = self.yhat_tr.shape[0]+self.lags
            _t3 = len(self.f)
            a.plot(range(_t1,_t2),self.yhat_tr[:,i],label='fitted (train)',color='C0')
            a.plot(range(_t2,_t3),self.yhat_test[:,i],label='fitted (test)',color='C1')
            a.legend()
        _mse1= self.mse['train']['total']
        _mse2= self.mse['test']['total']
        _mse3= self.mse['test(base)']['total']
        _title =\
        'dataset: {0} \n method: {1} \n mrate: {2:.2f}% \n interpolation:{3} \n epochs={4} \n number of filters={5} \n lags = {6} \n mse(train) = {7:.2f}, mse(test) = {8:.2f}, mse(test_base) = {9:.2f} \n'.\
        format(self.learner.dataset_name,self.learner.method,self.learner.mrate_total*100,self.learner.interpolation_method,self.learner.epochs,self.learner.nof_filters,self.learner.lags,_mse1,_mse2,_mse3)
        fig.suptitle(_title)
        fig.tight_layout()
        return fig
