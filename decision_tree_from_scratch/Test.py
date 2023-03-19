#Necessary imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from Decision_Tree import HomemadeDecisionTreeClassifier
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from Decision_Tree import HomemadeDecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn import tree
import time
from sklearn.datasets import load_linnerud
from sklearn.datasets import load_digits

data = load_digits()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# custom decision tree classifier performance

start = time.time()
clf = HomemadeDecisionTreeClassifier(max_depth=6)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
end = time.time()

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(f"The accuracy for the custom DT model is {acc} and the time taken is {end-start}")

clf.visualize_tree(feature_names=data.feature_names)

# sklearn decision tree classifier performance

start = time.time()
clf = tree.DecisionTreeClassifier(random_state=42, max_depth = 6)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
end = time.time()

acc = accuracy(y_test, predictions)
print(f"The accuracy for the sklearn DT model is {acc} and the time taken is {end-start}")

###################################################### Regression Task ###############################################################
data_reg = load_linnerud()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)

# Train a decision tree regressor on the training set & Evaluate the model on the testing set
start = time.time()
dt = HomemadeDecisionTreeRegressor(max_depth=3, min_samples_split=12)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
end = time.time()


mse = mean_squared_error(y_test, y_pred)
print(f"The MSE for the custom DT Regressor model is {mse} and the time taken is {end-start}")
#checking with the sklearn decision tree


r2 = r2_score(y_test, y_pred)
print(f"The R2_Score for the custom DT Regressor model is {r2} and the time taken is {end-start}")

#Check this against SKLearn DT Regressor
start = time.time()
clf = tree.DecisionTreeRegressor(max_depth=3, min_samples_split=12,random_state=40)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end = time.time()

mse = mean_squared_error(y_test, y_pred)
print(f"The MSE for the SKlearn DT model is {mse} and the time taken is {end-start}")

r2 = r2_score(y_test, y_pred)
print(f"The R2_Score for the Sklearn DT model is {r2} and the time taken is {end-start}")