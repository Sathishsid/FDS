import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create data frames
Train_Data = pd.DataFrame(X_train, columns=iris.feature_names)
Train_Data['target'] = y_train
Test_Data = pd.DataFrame(X_test, columns=iris.feature_names)
Test_Data['target'] = y_test
# Save the data frames as CSV files
Train_Data.to_csv('IrisTrainData.csv', index=False)
Test_Data.to_csv('IrisTestData.csv', index=False)
missing_values = Train_Data.isnull().sum().sum()
print("Number of missing values in Train_Data:", missing_values)
setosa_count = Test_Data[Test_Data['target'] == 0].shape[0]
total_samples = Test_Data.shape[0]
proportion = setosa_count / total_samples
print("Proportion of Setosa types in Test_Data:", proportion)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Create and train the K-Nearest Neighbor model
model_1 = KNeighborsClassifier(n_neighbors=2)
model_1.fit(Train_Data.drop('target', axis=1), Train_Data['target'])
# Make predictions on the test data
y_pred = model_1.predict(Test_Data.drop('target', axis=1))
# Calculate accuracy score
accuracy = accuracy_score(Test_Data['target'], y_pred)
print("Accuracy score of model_1:", accuracy)
misclassified_indices = Test_Data.index[Test_Data['target'] != y_pred]
print("Indices of misclassified samples:", misclassified_indices)
from sklearn.linear_model import LogisticRegression
# Create and train the logistic regression model
model_2 = LogisticRegression()
model_2.fit(Train_Data.drop('target', axis=1), Train_Data['target'])
# Make predictions on the test data
y_pred = model_2.predict(Test_Data.drop('target', axis=1))
# Calculate accuracy score
accuracy = accuracy_score(Test_Data['target'], y_pred)
print("Accuracy score of model_2:", accuracy)