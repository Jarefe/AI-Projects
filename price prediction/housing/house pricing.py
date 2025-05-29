# -----------------------------------------------
# House Price Prediction - End-to-End ML Pipeline
# -----------------------------------------------
# This script loads a dataset, performs data exploration and preprocessing,
# encodes categorical data, and applies several ML models to predict house prices.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ---------------------------
# Step 1: Load the Dataset
# ---------------------------

# Load Excel file into a pandas DataFrame
dataset = pd.read_excel('HousePricePrediction.xlsx')

# Display first 5 rows for a quick preview
print("First 5 rows of the dataset:")
print(dataset.head(5))

# Print dimensions: (rows, columns)
print("Dataset shape:", dataset.shape)

# ---------------------------
# Step 2: Identify Data Types
# ---------------------------

# Helps with preprocessing: separate columns by type

# Categorical columns (usually strings or objects)
object_cols = list(dataset.select_dtypes(include='object').columns)
print('Number of Categorical Variables:', len(object_cols))

# Integer columns
num_cols = list(dataset.select_dtypes(include='int').columns)
print('Number of Integer Variables:', len(num_cols))

# Float columns
fl_cols = list(dataset.select_dtypes(include='float').columns)
print('Number of Float Variables:', len(fl_cols))

# ---------------------------
# Step 3: Exploratory Data Analysis (EDA)
# ---------------------------

# Plot heatmap of correlations among numeric columns
plt.figure(figsize=(12, 6))
sns.heatmap(
    dataset.select_dtypes(include=['number']).corr(),
    cmap='BrBG',
    fmt='.2f',
    linewidths=2,
    annot=True
)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Plot number of unique categories in each categorical feature
unique_values = [dataset[col].nunique() for col in object_cols]
plt.figure(figsize=(10, 6))
sns.barplot(x=object_cols, y=unique_values)
plt.xticks(rotation=90)
plt.title('Unique Categories per Categorical Feature')
plt.show()

# Plot frequency distribution of categorical values
plt.figure(figsize=(18, 36))
plt.suptitle('Category Distributions', fontsize=16)
plt.subplots_adjust(hspace=0.5)

for idx, col in enumerate(object_cols, start=1):
    plt.subplot(11, 4, idx)
    sns.barplot(x=dataset[col].value_counts().index, y=dataset[col].value_counts().values)
    plt.xticks(rotation=90)
    plt.title(col)

plt.show()

# ---------------------------
# Step 4: Data Cleaning
# ---------------------------

# Drop non-useful columns (e.g., 'Id' is just a unique identifier)
if 'Id' in dataset.columns:
    dataset.drop(['Id'], axis=1, inplace=True)

# Fill missing target values ('SalePrice') with mean (simple imputation)
dataset['SalePrice'].fillna(dataset['SalePrice'].mean(), inplace=True)

# Drop any rows still containing missing values
filtered_dataset = dataset.dropna()

# Confirm no missing values remain
print("Missing values after cleaning:")
print(filtered_dataset.isnull().sum())

# ---------------------------
# Step 5: Encode Categorical Variables
# ---------------------------

# One-Hot Encoding turns categorical columns into numeric columns (one column per category)
object_cols = list(filtered_dataset.select_dtypes(include='object').columns)

print('Categorical columns to encode:')
print(object_cols)

# Initialize encoder
OH_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Apply encoding to categorical columns
OH_encoded = pd.DataFrame(OH_encoder.fit_transform(filtered_dataset[object_cols]))

# Match encoded DataFrame with original index for joining
OH_encoded.index = filtered_dataset.index
OH_encoded.columns = OH_encoder.get_feature_names_out()

# Remove original categorical columns and append encoded columns
df_final = filtered_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_encoded], axis=1)

# ---------------------------
# Step 6: Prepare for Model Training
# ---------------------------

# Separate target (what we want to predict) and features (inputs)
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split dataset into 80% training, 20% validation
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0
)

# ---------------------------
# Step 7: Train & Evaluate Models
# ---------------------------

# ---- Model 1: Support Vector Regression (SVR) ----
print("\nTraining Support Vector Regressor...")
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
Y_pred_svr = model_SVR.predict(X_valid)

print("SVR Mean Absolute Percentage Error:", mean_absolute_percentage_error(Y_valid, Y_pred_svr))

# ---- Model 2: Random Forest Regressor ----
print("\nTraining Random Forest Regressor...")
model_RFR = RandomForestRegressor(n_estimators=10, random_state=0)
model_RFR.fit(X_train, Y_train)
Y_pred_rfr = model_RFR.predict(X_valid)

print("Random Forest MAPE:", mean_absolute_percentage_error(Y_valid, Y_pred_rfr))

# ---- Model 3: Linear Regression ----
print("\nTraining Linear Regression...")
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred_lr = model_LR.predict(X_valid)

print("Linear Regression MAPE:", mean_absolute_percentage_error(Y_valid, Y_pred_lr))
