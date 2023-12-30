import random
import numpy as np
from models import *
import torch
import torch.nn as nn
import torch.optim as optim

## try to do it by operation

class TrainLinModel:
    def __init__(self, learning_rate = 0.01):
        self.w = [random.gauss(-5,5),random.gauss(-5,5)] #weights
        self.b = random.gauss(-5,5) 
        self.info = (self.w, self.b)
        
    def f(self,x):
        output = self.activation(self.w[0]*x + self.w[1]*x + self.b) # 
        return output
    
    # ReLU
    # def activation(self,x):
    #     return max(0,x)
    def activation(self,x):
        return x
        
    def descent(self, gradients, learning_rate):
        self.w[0] -= learning_rate*gradients[0]
        self.w[1] -= learning_rate*gradients[1]
        self.b -= learning_rate*gradients[2]
        
    
    def loss_func(self, output, answer):
        return (output-answer)**2
    
    def d_loss_func(self, data, output, answer):
        return (2*data)*(output-answer)
        
    def x_intercept(self,x0):
        return self.w[0]*x0/-self.w[1] + self.b/-self.w[1]
    
    def predict_func(self, x):
        return sum(self.w)*x + self.b

class TrainQuadModel():

    def __init__(self):
        super().__init__()
        self.w = [random.gauss(-5,5),random.gauss(-5,5)] #weights
        self.b = [random.gauss(-5,5), random.gauss(-5,5)] #biases
        self.info = (self.w, self.b)
        
    def z(self,x):
        return self.w[0]*x + self.b[0]
    
    def descent(self, gradients, learning_rate):
        self.w[0] -= learning_rate*gradients[0]
        self.w[1] -= learning_rate*gradients[1]
        self.b[0] -= learning_rate*gradients[2]
        self.b[1] -= learning_rate*gradients[3]
        
    def dloss_dw1(self, output, answer,x):
        return 2*(output-answer)*self.w[1]*(1-np.tanh(self.z(x))**2)*x
    
    def dloss_dw2(self, output, answer,x):
        return 2*(output-answer)*np.tanh(self.z(x))
    
    def dloss_db1(self,output,answer,x):
        return 2*self.w[1]*(output-answer)*(1-np.tanh(self.z(x))**2)
    
    def dloss_db2(self,output, answer,x):
        return 2*(output-answer)
    
    def x_intercept(self,x0):
        return self.w[0]*x0/-self.w[1] + self.b/-self.w[1]
    
    def func(self, x):
        return self.w[1] * np.tanh(self.w[0] * x + self.b[0]) + self.b[1]

class QuadraticNet(nn.Module):
    def __init__(self):
        super(QuadraticNet, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class PytorchQuadTrain:
    def __init__(self, epochs, learning_rate=0.0001):
        
    # Init: data, model, loss function, and optimizer
        
        self.data = QuadModel().torch_data(messy=True) 

        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.model = QuadraticNet()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        # Results
        self.losses = []
        self.epochs_list = []

        self.results = {"losses": self.losses,
                        "epochs": self.epochs_list}
    
    # Training loop    
    def train(self):

        x_data, y_data = self.data

        for epoch in range(self.epochs):

            # Forward pass
            predictions = self.model(x_data)

            # Calculate the loss
            loss = self.criterion(predictions, y_data)
            self.losses.append(loss.item())
            self.epochs_list.append(epoch + 1)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            self.model.eval()
            self.x_test = torch.linspace(torch.min(x_data), torch.max(x_data), 200).view(-1, 1)
            self.y_pred = self.model(self.x_test)
            
            print("Training complete")
            return self.x_test, self.y_pred


### to be done
        
# class PytorchPolyTrain(PolynomialModel):
#     def __init__(self, order, epochs, learning_rate=0.0001):
        
#         # inheriting data from the polynomial function
#         super().__init__()
#         self.order = order
#         self.data = self.torch_data

#     # Init: data, model, loss function, and optimizer
#         self.epochs = epochs
#         self.learning_rate = learning_rate
#         self.model = QuadraticNet()
#         self.criterion = nn.MSELoss()
#         self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
#         # Results
#         self.losses = []
#         self.epochs_list = []

#         self.results = { "losses": self.losses,
#                          "epochs": self.epochs_list}
    
#     # Training loop    
#     def train(self):

#         x_data, y_data = self.data

#         for epoch in range(self.epochs):

#             # Forward pass
#             predictions = self.model(x_data)

#             # Calculate the loss
#             loss = self.criterion(predictions, y_data)
#             self.losses.append(loss.item())
#             self.epochs_list.append(epoch + 1)

#             # Backward pass and optimization
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#         with torch.no_grad():
#             self.model.eval()
#             self.x_test = torch.linspace(torch.min(x_data), torch.max(x_data), 200).view(-1, 1)
#             self.y_pred = self.model(self.x_test)
            
#             print("Training complete")
#             return self.x_test, self.y_pred
