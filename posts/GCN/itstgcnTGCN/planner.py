import itertools
import time
import os 
import pandas as pd
import datetime
import torch_geometric_temporal 
from .learners import * 
from .utils import *
from .missing import * 

class PLNR():
    def __init__(self,plans,loader,dataset_name=None,simulation_results=None):
        self.plans = plans
        col = ['dataset', 'method', 'mrate', 'mtype', 'lags', 'nof_filters', 'inter_method', 'epoch', 'mse','calculation_time']
        self.loader = loader
        self.dataset_name = dataset_name
        self.simulation_results = pd.DataFrame(columns=col) if simulation_results is None else simulation_results 
    def record(self,method,mrate,mtype,lags,nof_filters,inter_method,epoch,mse,calculation_time):
        dct = {'dataset': self.dataset_name,
               'method': method,
               'mrate': mrate,
               'mtype': mtype, 
               'lags': lags,
               'nof_filters': nof_filters,
               'inter_method': inter_method,
               'epoch': epoch,
               'mse': mse,
               'calculation_time': calculation_time
              }
        simulation_result_new = pd.Series(dct).to_frame().transpose()
        self.simulation_results = pd.concat([self.simulation_results,simulation_result_new]).reset_index(drop=True)
    def save(self):
        if 'simulation_results' not in os.listdir(): 
            os.mkdir('simulation_results')
        fname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv")
        self.simulation_results.to_csv('./simulation_results/'+fname,index=False)    
        print("All results are stored in ./simulation_results/"+fname)
        
class PLNR_STGCN_RAND(PLNR):
    def simulate(self):
        for _ in range(self.plans['max_iteration']):  
            product_iterator = itertools.product(
                self.plans['method'], 
                self.plans['mrate'], 
                self.plans['lags'], 
                self.plans['nof_filters'], 
                self.plans['inter_method'],
                self.plans['epoch']
            )
            for prod_iter in product_iterator:
                method,mrate,lags,nof_filters,inter_method,epoch = prod_iter
                self.dataset = self.loader.get_dataset(lags=lags)
                train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(self.dataset, train_ratio=0.8)
                if mrate > 0: 
                    mtype = 'rand'
                    mindex = rand_mindex(train_dataset,mrate=mrate)
                    train_dataset = padding(train_dataset_miss = miss(train_dataset,mindex=mindex,mtype=mtype),interpolation_method=inter_method)
                elif mrate ==0: 
                    mtype = None
                    inter_method = None 
                if method == 'STGCN':
                    lrnr = StgcnLearner(train_dataset,dataset_name=self.dataset_name)
                elif method == 'IT-STGCN':
                    lrnr = ITStgcnLearner(train_dataset,dataset_name=self.dataset_name)
                t1 = time.time()
                lrnr.learn(filters=nof_filters,epoch=epoch)
                t2 = time.time()
                evtor = Evaluator(lrnr,train_dataset,test_dataset)
                evtor.calculate_mse()
                mse = evtor.mse['test']['total']
                calculation_time = t2-t1
                self.record(method,mrate,mtype,lags,nof_filters,inter_method,epoch,mse,calculation_time)
            print('{}/{} is done'.format(_+1,self.plans['max_iteration']))
        self.save()

class PLNR_STGCN_MANUAL(PLNR):
    def simulate(self,mindex,mtype):
        for _ in range(self.plans['max_iteration']):
            product_iterator = itertools.product(
                self.plans['method'], 
                self.plans['mindex'],
                self.plans['lags'],
                self.plans['nof_filters'],
                self.plans['inter_method'],
                self.plans['epoch']
            )
            for prod_iter in product_iterator:
                method,mrate,lags,nof_filters,inter_method,epoch = prod_iter
                self.dataset = self.loader.get_dataset(lags=lags)
                train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(self.dataset, train_ratio=0.8)
                mtype = mtype
                train_dataset = padding(train_dataset_miss = miss(train_dataset,mindex=mindex,mtype=mtype),interpolation_method=inter_method)
                if method == 'STGCN':
                    lrnr = StgcnLearner(train_dataset,dataset_name=self.dataset_name)
                elif method == 'IT-STGCN':
                    lrnr = ITStgcnLearner(train_dataset,dataset_name=self.dataset_name)
                t1 = time.time()
                lrnr.learn(filters=nof_filters,epoch=epoch)
                t2 = time.time()
                evtor = Evaluator(lrnr,train_dataset,test_dataset)
                evtor.calculate_mse()
                mse = evtor.mse['test']['total']
                mrate= lrnr.mrate_total
                calculation_time = t2-t1
                self.record(method,mrate,mtype,lags,nof_filters,inter_method,epoch,mse,calculation_time)
            print('{}/{} is done'.format(_+1,self.plans['max_iteration']))
        self.save()
        
class PLNR_GNAR_RAND(PLNR):
    def simulate(self):
        for _ in range(self.plans['max_iteration']):
            product_iterator = itertools.product(
                self.plans['mrate'],
                self.plans['lags'],
                self.plans['inter_method']
            )
            for prod_iter in product_iterator:
                mrate,lags,inter_method = prod_iter
                self.dataset = self.loader.get_dataset(lags=lags)
                train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(self.dataset, train_ratio=0.8)
                if mrate > 0: 
                    mtype = 'rand'
                    mindex = rand_mindex(train_dataset,mrate=mrate)
                    train_dataset = padding(train_dataset_miss = miss(train_dataset,mindex=mindex,mtype=mtype),interpolation_method=inter_method)
                elif mrate ==0: 
                    mtype = None
                    inter_method = None 
                method = 'GNAR'
                lrnr = GNARLearner(train_dataset,dataset_name=self.dataset_name)
                t1=time.time()
                lrnr.learn()
                t2=time.time()
                evtor = Evaluator(lrnr,train_dataset,test_dataset)
                evtor.calculate_mse()
                mse = evtor.mse['test']['total']
                nof_filters = None 
                epoch= None
                calculation_time = t2-t1
                self.record(method,mrate,mtype,lags,nof_filters,inter_method,epoch,mse,calculation_time)
            print('{}/{} is done'.format(_+1,self.plans['max_iteration']))
        self.save()
            
class PLNR_GNAR_MANUAL(PLNR):
    def simulate(self,mindex,mtype):
        for _ in range(self.plans['max_iteration']):
            product_iterator = itertools.product(
                self.plans['mindex'],
                self.plans['lags'],
                self.plans['inter_method']
            )
            for prod_iter in product_iterator:
                mrate,lags,inter_method = prod_iter
                self.dataset = self.loader.get_dataset(lags=lags)
                train_dataset, test_dataset = torch_geometric_temporal.signal.temporal_signal_split(self.dataset, train_ratio=0.8)
                mtype = mtype
                train_dataset = padding(train_dataset_miss = miss(train_dataset,mindex=mindex,mtype=mtype),interpolation_method=inter_method)
                method = 'GNAR'
                lrnr = GNARLearner(train_dataset,dataset_name=self.dataset_name)
                t1= time.time()
                lrnr.learn()
                t2= time.time()
                evtor = Evaluator(lrnr,train_dataset,test_dataset)
                evtor.calculate_mse()
                mse = evtor.mse['test']['total']
                nof_filters = None 
                epoch= None
                mrate= lrnr.mrate_total
                calculation_time = t2-t1
                self.record(method,mrate,mtype,lags,nof_filters,inter_method,epoch,mse,calculation_time)
            print('{}/{} is done'.format(_+1,self.plans['max_iteration']))                
        self.save()