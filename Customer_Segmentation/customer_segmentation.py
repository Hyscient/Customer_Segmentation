import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load the data from CSV files, limiting to the first 10000 rows
df1 = pd.read_csv('2019-Oct.csv', nrows=10000)
df2 = pd.read_csv('2019-Nov.csv', nrows=10000)

# Concatenate the two dataframes
data = pd.concat([df1, df2], ignore_index=True)

# Filter out rows that contain non-numeric values in the 'price' column
data = data[pd.to_numeric(data['price'], errors='coerce').notnull()]

# Convert the 'price' column to float
data['price'] = data['price'].astype(float)

data['price'] = pd.to_numeric(data['price'], errors='coerce')
data.dropna(subset=['price'], inplace=True)


# feature selection
X = data[['price']]

# standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# reduce dimensionality
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_std)

# cluster analysis
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(X_std)

# create cluster label column
data['Cluster'] = kmeans.labels_

# plot clusters
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), c=kmeans.labels_, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.title('Customer Segmentation based on K-means Clustering')
plt.show()

# analyze cluster characteristics
data.groupby(['Cluster']).mean()
