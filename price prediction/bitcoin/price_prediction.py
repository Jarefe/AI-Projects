# === Importing Required Libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# === Load and Preview Dataset ===
df = pd.read_csv('bitcoin.csv')

print(df.head())  # Show first few rows
print(df.shape)  # Print dataset dimensions
print(df.describe())  # Descriptive stats
print(df.info())  # Data types and non-null counts

# === Exploratory Data Analysis (EDA) ===

# Line plot of 'Close' price over time
plt.figure(figsize=(15, 5))
plt.plot(df['Close'], label='Close Price')
plt.title('Bitcoin Close Price Over Time', fontsize=15)
plt.ylabel('Price in USD')
plt.xlabel('Time (Index)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Check for duplicate columns (Adj Close vs Close)
same_close = df[df['Close'] == df['Adj Close']]
print(f"'Close' equals 'Adj Close' in {same_close.shape[0]} out of {df.shape[0]} rows.")

# Drop 'Adj Close' since it's identical to 'Close'
df.drop(columns=['Adj Close'], inplace=True)

# Check for missing data
print("Missing values per column:\n", df.isnull().sum())

# === Univariate Feature Distributions ===
features = ['Open', 'High', 'Low', 'Close']

# Histogram distribution with KDE for each feature
plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    sb.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Boxplot for detecting outliers
plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    sb.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# === Feature Engineering ===

# Convert Date to datetime and extract year/month/day
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

# Preview new features
print(df[['Date', 'year', 'month', 'day']].head())

# Group by year and plot average yearly values
data_grouped = df.groupby('year').mean(numeric_only=True)

plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    data_grouped[col].plot(kind='bar')
    plt.title(f'Yearly Average of {col}')
plt.tight_layout()
plt.show()

# Identify quarter-end months (March, June, September, December)
df['is_quarter_end'] = (df['month'] % 3 == 0).astype(int)

# Additional technical features
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']

# Target variable: 1 if tomorrow's close is higher than today's, else 0
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Check class balance
plt.figure(figsize=(6, 6))
plt.pie(df['target'].value_counts(), labels=['Down (0)', 'Up (1)'], autopct='%1.1f%%', colors=['red', 'green'])
plt.title('Target Class Distribution')
plt.show()

# === Correlation Matrix (Heatmap) ===
# Identify highly correlated features that might introduce redundancy
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr(numeric_only=True) > 0.9, annot=True, cbar=False)
plt.title("Feature Correlation > 0.9")
plt.show()

# === Prepare Features and Target ===
X = df[['open-close', 'low-high', 'is_quarter_end']]
y = df['target']

# Standardize feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Time-Series Aware Split ===
# First 70% for training, last 30% for validation
split_index = int(len(X_scaled) * 0.7)
X_train, X_valid = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_valid = y[:split_index], y[split_index:]

# === Model Training and Evaluation ===
models = [
    LogisticRegression(),
    SVC(kernel='poly', probability=True),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss')
]

for model in models:
    model.fit(X_train, y_train)

    print(f"{model.__class__.__name__} Performance:")
    train_score = metrics.roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_score = metrics.roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    print(f"  ➤ Training AUC:   {train_score:.4f}")
    print(f"  ➤ Validation AUC: {val_score:.4f}")
    print("-" * 40)

# High discrepancy in training AUC vs Validation AUC entails overfitting

# === Confusion Matrix for First Model ===
ConfusionMatrixDisplay.from_estimator(models[0], X_valid, y_valid)
plt.title(f"Confusion Matrix: {models[0].__class__.__name__}")
plt.show()
