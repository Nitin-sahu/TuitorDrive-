import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv(r"C:\Users\HP\Desktop\python project\ML Algorithms with Python Assignment (Data) (1).csv")

# Display basic info and check for missing values
print(data.info())
print(data.isnull().sum())

# Handle missing values (if any)
data = data.dropna()

# Descriptive statistics
print(data.describe())

# Visualization: Distribution of Insurance Charges
plt.figure(figsize=(8, 5))
sns.histplot(data['charges'], bins=30, kde=True)
plt.title("Distribution of Insurance Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()

# Visualization: Boxplot of Charges by Smoking Status
plt.figure(figsize=(8, 5))
sns.boxplot(x='smoker', y='charges', data=data)
plt.title("Boxplot of Charges by Smoking Status")
plt.xlabel("Smoker (No = 0, Yes = 1)")
plt.ylabel("Charges")
plt.show()

# Visualization: Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Visualization: Scatter Plot of BMI vs. Charges
plt.figure(figsize=(8, 5))
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=data, alpha=0.7)
plt.title("Scatter Plot of BMI vs. Charges")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show()

# Visualization: Bar Plot of Average Charges by Region
plt.figure(figsize=(8, 5))
sns.barplot(x='region', y='charges', data=data, estimator=np.mean)
plt.title("Average Insurance Charges by Region")
plt.xlabel("Region")
plt.ylabel("Average Charges")
plt.show()

# Encoding categorical variables
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']
target = 'charges'

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Splitting the dataset
X = data.drop(columns=[target])
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Model Training and Evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print("Training Performance:")
    print(f"R^2: {r2_score(y_train, y_pred_train):.3f}")
    print(f"RMSE: {mean_squared_error(y_train, y_pred_train, squared=False):.3f}")
    
    print("Testing Performance:")
    print(f"R^2: {r2_score(y_test, y_pred_test):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred_test, squared=False):.3f}")
    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred_test)
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title(f"Model: {model.__class__.__name__}")
    plt.show()

# Linear Regression
lr_model = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])
evaluate_model(lr_model, X_train, y_train, X_test, y_test)

# Random Forest Regressor
rf_model = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
evaluate_model(rf_model, X_train, y_train, X_test, y_test)
