import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("ObesityDataset.csv")
df.columns = df.columns.str.strip()
df_encoded = pd.get_dummies(df)

# Compute correlation matrix
correlation_matrix = df_encoded.corr()

# Display correlation with target variable only
print(correlation_matrix[['NObeyesdad']])