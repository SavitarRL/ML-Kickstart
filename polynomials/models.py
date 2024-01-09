import numpy as np
import random
import torch
   
class LinModel:
    def __init__(self, m= 2,c=4, num_points = 20):
        self.m = m
        self.c = c
        self.num_points = num_points

    def func(self,x):
        return self.m*x +self.c
        
    def true_data(self, messy = False):
        x_list = np.linspace(-5,5,self.num_points)
        data = self.func(x_list)
        answer_data = []
        
        for i in range(self.num_points):
            random_errors = 0
            if messy:
                random_errors = random.uniform(-2,2)
            # print(random_errors)
            answer_data.append((x_list[i], data[i]+random_errors))
        return answer_data           
    
class QuadModel:
    def __init__(self, a= 1,b=5,c=6):
        self.a = a
        self.b = b
        self.c = c
        
    def func(self,x):
        return self.a*x**2 + self.b*x+self.c

    def true_data(self, num_points = 1000, messy = False):
        x_list = np.linspace(-10,10,num_points)
        data = self.func(x_list)
        answer_data = []
        
        for i in range(num_points):
            random_errors = 0
            if messy:
                random_errors = random.uniform(-2,2)
            # print(random_errors)
            answer_data.append((x_list[i], data[i]+random_errors))
            # answer_data.append((x_list[i], data[i]))

        return answer_data
    
    def torch_data(self, num_points=200,messy=True):

        x_tensor = torch.linspace(-10,10,num_points).view(-1, 1)
        if messy:
            y_tensor = self.func(x_tensor) + torch.randn_like(x_tensor) * 3
        else:
            y_tensor = self.func(x_tensor)
        return x_tensor, y_tensor
    
class PolynomialModel:
    def __init__(self, order = 3):
        self.order = order
        self.coeffs = [random.uniform(-10,10) for _ in range(order)]

    def func(self,x):
        output = 0
        for n in range(self.order):
            output += self.coeffs[n] + x**n 

        return output

    def torch_data(self, num_points=200):
        x_tensor = torch.linspace(-10,10,num_points).view(-1, 1)
    
        y_tensor = self.func(x_tensor) + torch.randn_like(x_tensor) * 4
        
        return x_tensor, y_tensor
