import numpy as np

class oneLayerNN:
    def initializeZeros(self, m):
        W = np.zeros((m, 1))
        b = 0
        return W, b
    
    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s
    
    def propagate(self, W, b, X, Y, m):
        A = self.sigmoid(W.T @ X + b)
        cost = -(1/m) * ((Y @ np.log(A).T) + ((1 - Y) @ np.log(1 - A).T))
        dw = (1/m)*(X @ (A-Y).T)
        db = (1/m)*(np.sum(A-Y))
        cost = np.squeeze(cost)

        grads = {
            "dw" : dw,
            "db" : db
        }
        
        return grads, cost
    
    def fit(self, X, Y, epochs = 1000, learning_rate = .008 ):
        m = X.shape[1]
        W, b = self.initializeZeros(X.shape[0])

        for _ in range(epochs):
            grads , cost = self.propagate(W, b, X, Y, m)
            dw = grads["dw"]
            db = grads["db"]

            W -= (learning_rate * dw)
            b -= (learning_rate * db)

        params = {
            "W" : W,
            "b" : b
        }

        grads = {
            "dw" : dw,
            "db" : db
        }
        self.params = params
        return grads
    
    def predict(self, X):

        m = X.shape[1]
        W = self.params["W"]
        b = self.params["b"]
        Y = np.zeros((1, m))

        A = self.sigmoid(W.T @ X + b) 
        for i in range(m):
            Y[0,i] = 1 if A[0, i] > .5 else 0

        return Y


