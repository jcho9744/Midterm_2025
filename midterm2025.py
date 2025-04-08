# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Set random seed for reproducibility
np.random.seed(42)

# ------------------------------------------------------------------------------
# 2.1 Acquire, Clean, and Preprocess Data
# ------------------------------------------------------------------------------

# (a) Data Acquisition
# Load the dataset from the CSV file
df = pd.read_csv("bank_marketing.csv", sep=";")

print("Dataset loaded successfully. Shape:", df.shape)

print("\nMissing values:\n", df.isnull().sum())

df.replace("unknown", np.nan, inplace=True)
print("\nMissing values after replacing 'unknown' with NaN:\n", df.isnull().sum())

df.dropna(inplace=True)
print("\nShape after dropping missing values:", df.shape)

df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)