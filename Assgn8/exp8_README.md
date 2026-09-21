# Experiment 8: Clustering Human Activity Recognition Data using K-Means, DBSCAN, and Hierarchical Clustering

Student: S.S.Janane
Register Number: 3122247001024
Degree: M.Tech Integrated CSE(5 yrs)
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To implement and compare K-Means, DBSCAN, and Hierarchical Agglomerative Clustering on the Human Activity Recognition dataset, selecting cluster counts and density parameters through elbow, silhouette, and k-distance analysis, and evaluating cluster quality against the true activity labels.

## Dataset
- Name: Human Activity Recognition Using Smartphones (UCI HAR Dataset)
- Instances: 10299 samples (7352 train, 2947 test, combined for clustering)
- Features: 561 time and frequency domain features extracted from 3-axis accelerometer and gyroscope signals (50 Hz, 2.56s windows)
- Classes (used only for evaluation, not training): 6 activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)

## Methodology
- Preprocessing: missing value check, feature standardization using StandardScaler, PCA retaining 95 percent variance (104 components).
- Exploratory Data Analysis: activity class distribution, PCA and t-SNE 2D projections colored by true activity labels.
- K-Means: Elbow method (WCSS) and Silhouette score across k = 2 to 8 to select the best k.
- DBSCAN: k-distance graph to select eps, tuned alongside min_samples.
- Hierarchical Clustering: Agglomerative clustering with Ward linkage, dendrogram visualization on a sample of points.
- Evaluation: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index (internal metrics); Adjusted Rand Index and Normalized Mutual Information against true activity labels (external metrics); confusion matrix mapping clusters to activities.
- Comparison: bar charts of internal and external metrics across all three algorithms.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook exp8.ipynb
```
