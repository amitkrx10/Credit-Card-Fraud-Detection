import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Just checking how big the dataset is
print("Shape of dataset:", df.shape)

# Looking at first few rows to understand data
print(df.head())

# Basic info like data types and columns
print(df.info())

# Checking if any missing values are there
print(df.isnull().sum())

# Let’s see how many fraud and normal transactions we have
print(df['Class'].value_counts())

# Plotting fraud vs normal transactions
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()

# Checking how transaction amounts are distributed
plt.figure(figsize=(8,5))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.show()

# Let’s compare time pattern for fraud and normal transactions
plt.figure(figsize=(10,5))

sns.histplot(df[df['Class'] == 0]['Time'], bins=50, color='blue', label='Normal', alpha=0.6)
sns.histplot(df[df['Class'] == 1]['Time'], bins=50, color='red', label='Fraud', alpha=0.6)

plt.legend()
plt.title("Fraud vs Normal Transactions Over Time")
plt.show()

# Now checking correlation between features
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

print("EDA done, dataset understood 👍")