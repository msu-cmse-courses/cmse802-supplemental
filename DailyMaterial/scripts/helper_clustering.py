# Shin-Han Shiu
# Helper functions for Day 21 exercises

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import f1_score




def plot_class_cluster_2d(X, y, x0, x1, cluster_labels, title):
  '''Plot class and cluster labels in 2D
  Args:
    X (DataFrame): feature matrix
    y (Series): target class labels
    x0 (str): column name in X for x-axis
    x1 (str): column name in X for y-axis
    cluster_labels (ndarray): cluster labels
    title (str): title of the cluster label subplot
  Return:
    None
  '''
  fig, ax = plt.subplots(1, 2) 
  fig.set_size_inches(8, 4)

  # plot the class labels
  class0 = ax[0].scatter(X[x0][y==0], X[x1][y==0], c='b')
  class1 = ax[0].scatter(X[x0][y==1], X[x1][y==1], c='r')
  class2 = ax[0].scatter(X[x0][y==2], X[x1][y==2], c='c')
  ax[0].set_title('True Labels')
  ax[0].legend((class0, class1, class2), ("class0", "class1", "class2"))
  ax[0].set_xlim(-4, 4)
  ax[0].set_ylim(-4, 4)

  # plot the cluster labels
  clust0 = ax[1].scatter(X[x0][cluster_labels==0], 
                         X[x1][cluster_labels==0], c='b', alpha=0.5)
  clust1 = ax[1].scatter(X[x0][cluster_labels==1], 
                         X[x1][cluster_labels==1], c='r', alpha=0.5)
  clust2 = ax[1].scatter(X[x0][cluster_labels==2], 
                         X[x1][cluster_labels==2], c='c', alpha=0.5)
  ax[1].set_title(title)
  ax[1].legend((clust0, clust1, clust2), ("cluster0", "cluster1", "cluster2"))
  ax[1].set_xlim(-4, 4)
  ax[1].set_ylim(-4, 4)
  plt.show()


def plot_contingency(class_labels, cluster_labels):
  '''Plot the contingency matrix as a heatmap
  Args:
    class_labels (ndarray): true class labels
    cluster_labels (ndarray): original cluster labels
  Return:
    c_df (DataFrame): contingency matrix as a dataframe
  '''
  
  c_matrix = contingency_matrix(class_labels, cluster_labels)

  # convert contingency matrix to a dataframe
  c_df = pd.DataFrame(c_matrix,
                      index=["class0", "class1", "class2"],
                      columns=["cluster0", "cluster1", "cluster2"])
  # plot heatmap
  plt.figure(figsize=(4, 3))
  sns.heatmap(c_df, annot=True, cmap='Blues')
  plt.show()

  return c_df


def get_f1(class_labels, cluster_labels, cluster_names_order, verbose=0):
  '''Calcuilate F1 score based on class labels and sorted cluster labels
  Args:
    class_labels (ndarray): true class labels
    cluster_labels (ndarray): original cluster labels
    cluster_names_order (list): cluster names in order of corresponding class
    verbose (int): print intermediate results if 1
  Return:
    f1 (float): F1 score
  '''
  # comment
  cluster_idx_sorted = \
    [int(cluster_name[-1:]) for cluster_name in cluster_names_order]
  if verbose:
    print('\ncluster_names_order', cluster_names_order)
    print('\ncluster_idx_sorted', cluster_idx_sorted)

  # comment
  cluster_labels_sorted = [cluster_idx_sorted.index(i) for i in cluster_labels]
  if verbose:
    df = pd.DataFrame({'class_labels': class_labels, 
                       'cluster_labels': cluster_labels, 
                       'cluster_labels_sorted': cluster_labels_sorted})
    print("\n", df.sample(10))

  f1 = f1_score(class_labels, cluster_labels_sorted, average='weighted')
  print("\nF1 score: %.2f" % f1)

  return f1
