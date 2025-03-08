import numpy as np
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor

class SGDLinearRegression:
    def __init__(self,learning_rate=0.01,epochs=100,batch_size=32):
        self.learning_rate= learning_rate
        self.epochs=epochs
        self.batch_size=batch_size
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self,x,y):
        # Initialize parameters
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #SGD optimization
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            # Mini batch Gradient Descent
            for i in range(0,n_samples,self.batch_size):
                #get mini batch
                end = min(i+self.batch_size,n_samples)
                x_batch = x_shuffled[i:end]
                y_batch = y_shuffled[i:end]

                batch_size = x_batch.shape[0]

                #prediction
                y_pred = np.dot(x_batch,self.weights) + self.bias
                
                #compute loss
                dw = (1/batch_size)*np.dot(x_batch.T,(y_pred-y_batch))
                db = (1/batch_size)*np.sum(y_pred-y_batch)

                #update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # Compute loss on the full  dataset for monitoring
            y_pred_full = np.dot(x,self.weights) + self.bias
            loss = np.mean((y_pred_full - y) ** 2)
            self.losses.append(loss)

            if epoch %10 ==0:
                print(f'Epoch {epoch+1}, Loss: {loss:.3f}')
    def predict(self,x):
            return np.dot(x,self.weights)+self.bias


#Load the California housing dataset
df = fetch_california_housing()
x, y = df.data, df.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

## SGD fro0m scratch

print("Training SGD Linear Regression from scratch...")
model_SGD_scratch = SGDLinearRegression(learning_rate=0.01,epochs=50,batch_size=32)
model_SGD_scratch.fit(x_train_scaled,y_train)

y_pred_scratch = model_SGD_scratch.predict(x_test_scaled)

print("\nSGD Linear Regression from Scratch:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_scratch)}")
print(f"R² Score: {r2_score(y_test, y_pred_scratch)}")

# -------------------- Scikit-learn SGD Implementation --------------------
print("\nTraining SGD Linear Regression with scikit-learn...")
# Train the model
model_SDG_Sklear = SGDRegressor(
    max_iter=1000, 
    tol=1e-3, 
    penalty=None, 
    eta0=0.01, 
    learning_rate='constant',
    random_state=42
)

model_SDG_Sklear.fit(x_train_scaled,y_train)

y_pred_sklearn = model_SDG_Sklear.predict(x_test_scaled)


print("\nSGD Linear Regression using Scikit-learn:")
print(f"Coefficients: {model_SDG_Sklear.coef_}")
print(f"Intercept: {model_SDG_Sklear.intercept_}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_sklearn)}")
print(f"R² Score: {r2_score(y_test, y_pred_sklearn)}")