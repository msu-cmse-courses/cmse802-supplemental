# Shin-Han Shiu
# 3/25/2024
# Helper functions for Day 23 exercises

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from torcheval.metrics.functional import multiclass_f1_score



def get_subsets(data_dir, train_valid_size=200):
  '''Get training, validation, and test data
  Args:
    data_dir (str or Path): the directory to store the data
    train_valid_size (int): the number of images for training and validation
  Return:
    X_train (torch.Tensor): training images, defined by train_valid_size
    X_valid (torch.Tensor): validation images, defined by train_valid_size
    X_test (torch.Tensor): test images, all 6000 images per class
    y_train (torch.Tensor): training labels
    y_valid (torch.Tensor): validation labels
    y_test (torch.Tensor): test labels
    class_names (list): the names of the classes
  '''

  fashion_train_data =  datasets.FashionMNIST(root=data_dir, train=True, 
                download=True, transform=transforms.ToTensor())
  fashion_test_data  =  datasets.FashionMNIST(root=data_dir, train=False, 
                download=True, transform=None)

  class_names = fashion_train_data.classes

  # features and labels for the full training dataset
  X_train_all = fashion_train_data.data
  y_train_all = fashion_train_data.targets

  X_test_all = fashion_test_data.data
  y_test_all = fashion_test_data.targets



  # populate selected images
  X_train_list = []
  X_valid_list = []
  X_test_list  = []

  # populate selected labels
  y_train_list = []
  y_valid_list = []
  y_test_list  = []

  # go through each class
  for class_i in range(10):
    # get labels and features of class_i
    y_train_i = y_train_all[y_train_all == class_i]
    X_train_i = X_train_all[y_train_all == class_i]

    # test set
    y_test_i = y_test_all[y_test_all == class_i]
    X_test_i = X_test_all[y_test_all == class_i]

    # append the first train_valid_size to training set and the next 
    # train_valid_size to validation set, also normalize the features
    X_train_list.append(X_train_i[:train_valid_size]/255)
    X_valid_list.append(X_train_i[train_valid_size:train_valid_size*2]/255)
    y_train_list.append(y_train_i[:train_valid_size])
    y_valid_list.append(y_train_i[train_valid_size:train_valid_size*2])

    # deal with testing data
    X_test_list.append(X_test_i/255)
    y_test_list.append(y_test_i)

  # concatenate images in lists into tensor of shape (2000, ...)
  X_train = torch.cat(X_train_list, dim=0)
  X_valid = torch.cat(X_valid_list, dim=0)
  y_train = torch.cat(y_train_list, dim=0)
  y_valid = torch.cat(y_valid_list, dim=0)
  X_test  = torch.cat(X_test_list, dim=0)
  y_test  = torch.cat(y_test_list, dim=0)

  return X_train, X_valid, X_test, y_train, y_valid, y_test, class_names


def train(net, X_train, y_train, X_valid, y_valid, epochs=100, lr=0.01, wd=0.0,
          model_name=""):
  '''Train the artificial neural network
  Args:
    net (torch.nn.Module): the neural network to be trained
    X_train (torch.Tensor): the input data for training
    y_train (torch.Tensor): the input labels for training
    X_valid (torch.Tensor): the input data for validation
    y_valid (torch.Tensor): the input labels for validation
    epochs (int): the number of epochs, default 100
    lr (float): the learning rate, default 0.01
    wd (float): the weight decay, default 0.0
    model_name (str): the name of the model to be saved, if empty, no save
  Returns:
    epoch_loss (list): the loss after each epoch
    epoch_f1 (list): the f1 score after each epoch
  '''
  # pick a metric to measure the goodness of fit: negative log likelihood loss
  criterion = nn.NLLLoss()

  # Pick a training algorithm that iteratively minimizes the loss
  # Here, we pick stochastic gradient descent from torch.optim
  optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

  # list to store the loss and f1 after each epoch
  epoch_loss = []
  epoch_f1 = []

  # start training, update weights and bias.
  net.train()

  # go through the epochs
  for e in tqdm(range(epochs)):
    # clear previously stored gradients if any
    optimizer.zero_grad()

    # compute the output of the network given the input data
    y_train_pred = net(X_train)
    y_valid_pred = net(X_valid)

    # compute the difference between the predicted and true labels
    train_loss = criterion(y_train_pred, y_train)
    valid_loss = criterion(y_valid_pred, y_valid)

    # store the epoch and loss info
    epoch_loss.append([e, train_loss, valid_loss])

    # compute the multiclass f1 score for training and validation data
    train_f1 = multiclass_f1_score(y_train_pred.argmax(dim=1), y_train,
                                   num_classes=10, average='macro')
    valid_f1 = multiclass_f1_score(y_valid_pred.argmax(dim=1), y_valid,
                                   num_classes=10, average='macro')

    # store the epoch and f1 score
    epoch_f1.append([e, train_f1, valid_f1])

    # compute the gradient of the loss function with respect to the parameters
    # based on the training loss **NOT** validation loss
    train_loss.backward()

    # perform a gradient descent update
    optimizer.step()

  # save the model
  if model_name != "":
    torch.save(net, model_name)

  return epoch_loss, epoch_f1


def plot_train_valid_scores(epoch_loss, epoch_f1):
  '''Plot the training and validation losses and F1 over epochs
  Args:
    epoch_loss (list): the loss after each epoch
    epoch_f1 (list): the f1 score after each epoch
  '''
  
  epochs = [i[0] for i in epoch_loss]

  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
  for idx, scores in enumerate([epoch_loss, epoch_f1]):
    score_type = "loss" if idx == 0 else "f1"
    train_scores = [i[1].item() for i in scores]
    valid_scores = [i[2].item() for i in scores]

    ax[idx].plot(epochs, train_scores, label=f'Train {score_type}', c='b', 
                  alpha=0.3)
    ax[idx].plot(epochs, valid_scores, label=f'Valid {score_type}', c='r', 
                  alpha=0.3)
    ax[idx].set_xlabel('Epochs')
    ax[idx].set_ylabel(score_type)
    ax[idx].grid(True)

    x_text = 0
    scores_all = train_scores + valid_scores
    score_range = np.max(scores_all)-np.min(scores_all)
    y_text = (score_range)/2+np.min(scores_all)
    ax[idx].text(x_text, y_text, 
                 f"Final {score_type} train: {train_scores[-1]:.4f}",)
    ax[idx].text(x_text, y_text-score_range*0.1,
                 f"Final {score_type} valid: {valid_scores[-1]:.4f}")
    ax[idx].legend()
  plt.tight_layout()
  plt.show()