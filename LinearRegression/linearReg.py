import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regresion From Scratch

class MyLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iteration = iterations
        self.weights = None
        self.bias = None

    def fit(self,x,y):
        #initialize patrameters
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iteration):
            y_predicted = np.dot(x, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self,x):
        y_approximated = np.dot(x, self.weights) + self.bias
        return y_approximated

#Load CAlifornia housing Dataset
df = fetch_california_housing()
x,y = df.data, df.target

# For simplicity, let's use only the first feature (median income)
x_simple = x[:,0].reshape(-1,1)

#split data 
X_train, X_test, y_train, y_test = train_test_split(x_simple,y,test_size=0.2,random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model_scratch =MyLinearRegression(learning_rate=0.01,iterations=1000)
model_scratch.fit(X_train,y_train)

model_sklearn = LinearRegression()
model_sklearn.fit(X_train,y_train)

y_pred_scratch = model_scratch.predict(X_test)
y_pred_sklearn = model_sklearn.predict(X_test)

print("Linear Regression from Scratch:")
print(f"Weights: {model_scratch.weights}, Bias: {model_scratch.bias}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_scratch)}")
print(f"R² Score: {r2_score(y_test, y_pred_scratch)}")

print("/////////////////////////////////////////") 

print("\nLinear Regression using Scikit-learn:")
print(f"Weights: {model_sklearn.coef_}, Bias: {model_sklearn.intercept_}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_sklearn)}")
print(f"R² Score: {r2_score(y_test, y_pred_sklearn)}")


# -------------------- Multiple Feature Example --------------------
# Using all features from the California Housing dataset
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler_multi = StandardScaler()
X_multi_train_scaled = scaler_multi.fit_transform(X_multi_train)
X_multi_test_scaled = scaler_multi.transform(X_multi_test)

# Train scikit-learn model with multiple features
model_multi = LinearRegression()
model_multi.fit(X_multi_train_scaled, y_multi_train)

# Make predictions
y_multi_pred = model_multi.predict(X_multi_test_scaled)
print("/////////////////////////////////////////") 
print("\nMultiple Feature Linear Regression (Scikit-learn):")
print(f"Coefficients: {model_multi.coef_}")
print(f"Intercept: {model_multi.intercept_}")
print(f"Mean Squared Error: {mean_squared_error(y_multi_test, y_multi_pred)}")
print(f"R² Score: {r2_score(y_multi_test, y_multi_pred)}")