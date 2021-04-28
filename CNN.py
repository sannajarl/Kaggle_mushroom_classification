# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:23:11 2021

@author: SJARL

9 different types of mushrooms
"""

import torch
from torch import nn
import torch.optim as optim
from  torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import numpy as np
import os
from sklearn import metrics
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CNN(nn.Module):
    def __init__(self, in_channels, nr_classes):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = torch.nn.Linear(3 * 32 * 32, 2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   
        out = self.fc(out)
        return out
        
        
def load_data(data_folder):
    mushroom_rgb = []
    labels = []
    label_val = 0
    mushroom_folders = [os.path.join(data_folder, o) for o in os.listdir(data_folder)]
    for folder in mushroom_folders:
        imgs_path = [os.path.join(folder + '/', s) for s in os.listdir(folder)] # inne i varje svampmapp
        labels = labels + [label_val]*len(imgs_path)
        label_val += 1
        for img in imgs_path:
            mushroom_img = np.array(Image.open(img).convert('RGB'))
            mushroom_img = mushroom_img.astype('float32')
            mushroom_rgb.append(mushroom_img)
    x_train, x_test, y_train, y_test = train_test_split(mushroom_rgb, labels, test_size=0.85)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.8)
    trainset = create_dataset(x_train, y_train)
    valset = create_dataset(x_val, y_val)
    testset = create_dataset(x_test, y_test)
    return trainset, valset, testset

def create_dataset(x, y):
    torch_x = torch.tensor(x)
    torch_y = torch.tensor(y, dtype=torch.long)
    t_dataset = TensorDataset(torch_x, torch_y)
    return t_dataset
    
def evaluate_model(val_data_loader, valset, model, loss_fn):
    losses = []
    prob = torch.tensor([])
    n_correct = 0
    with torch.no_grad():
        for b_x, b_y in val_data_loader:
            pred = model.forward(b_x)
            loss = loss_fn(pred, b_y)
            losses.append(loss.item())
            prob = torch.cat((prob, pred), 0)
            n_correct += torch.sum(pred.argmax(dim=1) == b_y).item()
        val_accuracy = n_correct/len(valset)
        val_avg_loss = sum(losses)/len(losses)
    return val_accuracy, val_avg_loss, prob

def train_model(model, nEpochs, trainset, training_loader, loss_fn, optimizer):
    for epoch in range(nEpochs):
        losses = []
        n_correct = 0
        for b_x, b_y in training_loader:
            # Compute predictions and losses
            pred = model.forward(b_x)
            loss = loss_fn(pred, b_y.long())
            losses.append(loss.item())
        
            # Count number of correct predictions
            # hard_preds = pred.argmax(dim=1)
            n_correct += torch.sum(pred.argmax(dim=1) == b_y).item()
    
            # Backpropagate
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()   
            
        # Compute accuracy and loss in the entire training set
        train_accuracy = n_correct/len(trainset)
        train_avg_loss = sum(losses)/len(losses)    
            
        # Display metrics
        display_str = 'Epoch {} '
        display_str += '\tLoss: {:.3f} '
        display_str += '\tAccuracy: {:.2f} '
        print(display_str.format(epoch, train_avg_loss, train_accuracy))
        
    
if __name__ == "__main__":
    data_folder = '/chalmers/users/sannaja/Documents/Kaggle_mushroom_classification/Mushrooms/'
    images, labels = load_data(data_folder)
    model = CNN()
    trainset, valset, testset = load_data(data_folder)
    