"""
This is the same implementation of the original nohidden NN but using the pytorch lib
"""

import torch, numpy as np

class oneLayerNN:
    def initZeros(self, m):
        W = torch.zeros((m,1), requires_grad= True)
        b = torch.tensor([[0.]],requires_grad= True)
        return W, b
    
    def propagate(self, W: torch.Tensor , b: torch.Tensor, X, Y, m):
        Z = torch.matmul(W.T, X) + b
        A = torch.sigmoid(Z)
        cost = -(1/m) * ((torch.matmul(Y, torch.log(A).T)) + (torch.matmul((1 - Y) ,torch.log(1 - A).T)))
        cost = cost.squeeze()
        cost.backward()

        grads = {
            'dw' : W.grad,
            'db' : b.grad
        }

        return grads, cost
    
    def fit(self, X, Y, epochs = 1000, learning_rate = .008):
        m = X.shape[1]
        X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
        W, b = self.initZeros(X.shape[0])
        for _ in range(epochs):
            grads, cost = self.propagate(W, b, X, Y, m)
            if _ % 100 == 0:
                print(f"Avg cost in the iter {_} is: {cost:.3f} ")
            with torch.no_grad():
                W -= (learning_rate * grads["dw"])
                b -= (learning_rate * grads["db"])
            W.grad.zero_()
            b.grad.zero_()
        
        params = {
            'W' : W,
            'b' : b
        }
        
        self.params = params


    def predict(self, X):
        m = X.shape[1]
        X= torch.from_numpy(X).float()
        W = self.params["W"]
        b = self.params["b"]
        Y = np.zeros((1, m))

        A = torch.sigmoid(torch.matmul(W.T, X) + b) 
        for i in range(m):
            Y[0,i] = 1 if A[0, i] > .5 else 0

        return Y

