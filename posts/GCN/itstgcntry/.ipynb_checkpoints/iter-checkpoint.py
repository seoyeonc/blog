# torch
import torch

class standard_iter:
    def __init__(self):
        self.num_model = 1
        
    def do_iter(self,train_dataset,model,optimizer):
        for t, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).reshape(-1)
            cost = torch.mean((y_hat-snapshot.y)**2)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            
class accumulated_iter_first:
    def __init__(self):
        self.num_model = 1
        
    def do_iter(self,train_dataset,model,optimizer):
        cost = 0
        for t, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).reshape(-1)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (t+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        
class accumulated_iter_first_grad:
    def __init__(self):
        self.num_model = 1
        
    def do_iter(self,train_dataset,model,optimizer):
        cost = 0
        for t, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).reshape(-1)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (t+1)
        cost.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
            
class accumulated_iter_second:
    def __init__(self):
        self.num_model = 2
        
    def do_iter(self,train_dataset,model,optimizer):
        cost = 0
        self.h, self.c = None, None
        for t, snapshot in enumerate(train_dataset):
            y_hat, self.h, self.c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, self.h, self.c)
            y_hat = y_hat.reshape(-1)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (t+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

class accumulated_iter_third:
    def __init__(self):
        self.num_model = 3
        
    def do_iter(self,train_dataset,model,optimizer):
        cost = 0
        self.hidden_state = None
        for t, snapshot in enumerate(train_dataset):
            y_hat, self.hidden_state = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr,self.hidden_state)
            y_hat = y_hat.reshape(-1)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (t+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()