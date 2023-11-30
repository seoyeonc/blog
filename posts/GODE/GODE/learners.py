import numpy as np
import pandas as pd

# pyod
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.cblof import CBLOF
from pyod.models.ocsvm import OCSVM
from pyod.models.mcd import MCD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.abod import ABOD
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.sos import SOS
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.lscp import LSCP

# rpy2
import tqdm
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector 

from haversine import haversine

from .utils import Conf_matrx

def make_w(n):
    w = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j :
                w[i,j] = 0
            elif np.abs(i-j) <= 1 : 
                w[i,j] = 1
    return w

def eigen(W):
    d = W.sum(axis = 1)
    D = np.diag(d)
    L = np.diag(1 / np.sqrt(d)) @ (D - W) @ np.diag(1 / np.sqrt(d))
    lamb, Psi = np.linalg.eigh(L)
    return lamb, Psi 
    
def get_distance_matrix(x, y, n, method, theta=1, beta=0.5, kappa=4000):
    D = np.zeros([n,n])
    locations = np.stack([x, y],axis = 1)
    for i in tqdm.tqdm(range(n)):
        for j in range(i, n):
            if method == 'Euclidean':
                D[i, j] = np.linalg.norm(locations[i] - locations[j])
            elif method == 'Haversine':
                D[i, j] = haversine(locations[i], locations[j])
            else:
                pass      
    D = D + D.T
    dist = np.where(D < kappa,D,0)
    W = np.exp(-(dist/theta)**2)
    return W


class Linear:
    def __init__(self, df):
        self.df = df
        self.n = len(df)
    def fit(self,sd=20): # fit with ebayesthresh
        self.W = make_w(self.n)
        self.lamb, self.Psi = eigen(self.W)
        self.ybar = self.Psi.T @ self.df.y # fbar := graph fourier transform of f
        self.power = self.ybar**2 
        ebayesthresh = importr('EbayesThresh').ebayesthresh
        self.power_threshed=np.array(ebayesthresh(FloatVector(self.power),sd=sd))
        self.ybar_threshed = np.where(self.power_threshed>0,self.ybar,0)
        self.yhat = self.Psi@self.ybar_threshed
    def __call__(self, df):
        x = df.x.to_numpy()
        y = df.y.to_numpy()
        yhat = self.yhat
        return {'x':x, 'y':y, 'yhat':yhat} 
    
class Orbit:
    def __init__(self,df):
        self.df = df
        self.n = len(self.df)
    def fit(self,sd=5,method = 'Euclidean'): # fit with ebayesthresh
        self.W = get_distance_matrix(self.df.x, self.df.y, self.n, method, theta=1, beta=0.5, kappa=4000)
        self.lamb, self.Psi = eigen(self.W)
        self.fbar = self.Psi.T @ self.df.f # fbar := graph fourier transform of f
        self.power = self.fbar**2 
        ebayesthresh = importr('EbayesThresh').ebayesthresh
        self.power_threshed=np.array(ebayesthresh(FloatVector(self.power),sd=sd))
        self.fbar_threshed = np.where(self.power_threshed>0,self.fbar,0)
        self.fhat = self.Psi@self.fbar_threshed
    def __call__(self, df):
        x = df.x.to_numpy()
        y = df.y.to_numpy()
        f1 = df.f1.to_numpy()
        f = df.f.to_numpy()
        fhat = self.fhat
        return {'x':x, 'y':y,  'f1':f1, 'f':f,'fhat':fhat} 

class BUNNY:
    def __init__(self,df,W):
        self.df = df 
        self.n = len(self.df)
        self.W = W
    def fit(self,sd=5): # fit with ebayesthresh
        self.lamb, self.Psi = eigen(self.W)
        self.fbar = self.Psi.T @ self.df.f # fbar := graph fourier transform of f
        self.power = self.fbar**2 
        ebayesthresh = importr('EbayesThresh').ebayesthresh
        self.power_threshed=np.array(ebayesthresh(FloatVector(self.power),sd=sd))
        self.fbar_threshed = np.where(self.power_threshed>0,self.fbar,0)
        self.fhat = self.Psi@self.fbar_threshed
    def __call__(self, df):
        x = df.x.to_numpy()
        y = df.y.to_numpy()
        z = df.z.to_numpy()
        f1 = df.f1.to_numpy()
        f = df.f.to_numpy()
        fhat = self.fhat
        return {'x':x, 'y':y, 'z':z, 'f1':f1, 'f':f, 'fhat':fhat} 
        
class Earthquake:
    def __init__(self,df):
        self.df = df
        self.n = len(self.df)    
    def fit(self,sd=5, method = 'Haversine'):
        self.W = get_distance_matrix(self.df.x, self.df.y, self.n, method, theta=1, beta=0.5, kappa=4000)
        self.lamb, self.Psi = eigen(self.W)
        self.fbar = self.Psi.T @ self.df.f # fbar := graph fourier transform of f
        self.power = self.fbar**2 
        ebayesthresh = importr('EbayesThresh').ebayesthresh
        self.power_threshed=np.array(ebayesthresh(FloatVector(self.power),sd=sd))
        self.fbar_threshed = np.where(self.power_threshed>0,self.fbar,0)
        self.fhat = self.Psi@self.fbar_threshed
    def __call__(self, df):
        x = df.x.to_numpy()
        y = df.y.to_numpy()
        f = df.f.to_numpy()
        fhat = self.fhat
        return {'x':x, 'y':y, 'f':f,'fhat':fhat} 
        






