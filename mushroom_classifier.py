# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 20:52:28 2021
Classifying mushrooms, edible vs poisonous
labels: edible vs poisonous
cap shape: 6
cap surface: 4
cap color: 10
bruises: 2
odor: 9
gill attachment: 3
gill spacing: 3
gill size: 2
gill color: 12
stalk shape: 2
stalk root: 7
stalk surface above: 4
stalk surface below: 4
stalk color above: 9
stalk color below: 9
veil type: 2
veil color: 4
ring number: 3
ring type: 8
spore print: 9
population: 6
habitat: 7

@author: SJARL
"""
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from  torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
# import numpy as np
# import os
# from sklearn import metrics
# from sklearn.preprocessing import normalize
from sklearn import preprocessing

class Mushroom_Network(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, nr_classes):
        super().__init__()
        # self.bn1 = nn.BatchNorm1d(input_size, eps = 1e-5, momentum = 0.1)
        self.fc1 = nn.Linear(input_size, hidden1_size) #bias included unless stated false
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden1_size, eps = 1e-5, momentum = 0.1)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(hidden2_size, eps = 1e-5, momentum = 0.1)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.relu3 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(hidden3_size, eps = 1e-5, momentum = 0.1)
        self.fc4 = nn.Linear(hidden3_size, nr_classes)
        self.softmax = nn.Softmax(-1)
        
    def forward(self, X):
        # out = self.bn1(X)
        out = self.fc1(X)
        out = self.relu1(out)
        out = self.bn2(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.bn3(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.bn4(out)
        out = self.fc4(out)
        out = self.softmax(out)
        
        return out
    
def create_dataset(x, y):
    torch_x = torch.tensor(x)
    torch_y = torch.tensor(y, dtype=torch.long)
    t_dataset = TensorDataset(torch_x, torch_y)
    return t_dataset

def make_categorical(dataset, le): # takes in a dataframe, 
    headers = dataset.columns.values.tolist()
    categorical_list = []
    for item in headers:
        if item == 'class':
            #labels 
            labels = dataset[item].tolist()
            le.fit(labels)
            labels_cat = le.transform(labels)
        else:
            #features
            feat = dataset[item]
            le.fit(feat)
            categorical_feat = le.transform(feat)
            categorical_list.append(categorical_feat.astype('float32'))
    transpose_categorical = [[row[i] for row in categorical_list] for i in range(len(categorical_list[0]))]
    return transpose_categorical, labels_cat
    
def preprocess(dataset):
    le = preprocessing.LabelEncoder()
    X, labels = make_categorical(dataset, le)
    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
    # convert to torch tensor datasets
    torch_x = torch.tensor(x_train)
    torch_y = torch.tensor(y_train, dtype=torch.long)
    torch_test_x = torch.tensor(x_test)
    torch_test_y = torch.tensor(y_test, dtype=torch.long)
    t_dataset = TensorDataset(torch_x, torch_y)
    # split the train data into train and validation set
    trainset, valset = torch.utils.data.random_split(t_dataset, [int(len(t_dataset)*0.6875)+1, int(len(t_dataset)*0.3125)])
    testset = TensorDataset(torch_test_x, torch_test_y) 
    return trainset, valset, testset, [[x_train], [y_train]] 
    
    
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
            
            # hard_preds = pred.argmax(dim=1)
            n_correct += torch.sum(pred.argmax(dim=1) == b_y).item()
        val_accuracy = n_correct/len(valset)
        val_avg_loss = sum(losses)/len(losses)
    return val_accuracy, val_avg_loss, prob

def train_model(model, nEpochs, trainset, training_loader, valset, validation_loader, loss_fn, optimizer):
    train_loss, val_loss = [], []
    for epoch in range(nEpochs):
        losses = []
        n_correct = 0
        for b_x, b_y in training_loader:
            # Compute predictions and losses
            pred = model.forward(b_x)
            loss = loss_fn(pred, b_y.long())
            losses.append(loss.item())
        
            # Count number of correct predictions
            n_correct += torch.sum(pred.argmax(dim=1) == b_y).item()
    
            # Backpropagate
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()   
            
        # Compute accuracy and loss in the entire training set
        train_accuracy = n_correct/len(trainset)
        train_avg_loss = sum(losses)/len(losses)    
            
        # Compute accuracy and loss in the entire validation set
        val_accuracy, val_avg_loss, prob = evaluate_model(validation_loader, valset, model, loss_fn)
            
        # Display metrics
        display_str = 'Epoch {} '
        display_str += '\tLoss: {:.3f} '
        display_str += '\tLoss (val): {:.3f}'
        display_str += '\tAccuracy: {:.2f} '
        display_str += '\tAccuracy (val): {:.2f}'
        print(display_str.format(epoch, train_avg_loss, val_avg_loss, train_accuracy, val_accuracy))
        train_loss.append(train_avg_loss)
        val_loss.append(val_avg_loss)
    return train_loss, val_loss

def plot_losses(train_loss, val_loss):
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss)
    plt.plot(range(epochs), val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(["training", "Validation"], loc = "upper right")
    plt.title('Loss curves')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv(r'C:\Users\sjarl\Documents\kaggle_mushrooms\mushrooms.csv')
    
    input_size = 22
    hidden1_size = 10
    hidden2_size = 20
    hidden3_size = 30
    nr_classes = 2 
    nEpochs = 50
    
    model = Mushroom_Network(input_size, hidden1_size, hidden2_size, hidden3_size, nr_classes)
    
    # Train network
    trainset, valset, testset, [[x_train], [y_train]] = preprocess(data)
    training_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(valset, batch_size=32, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss = train_model(model, nEpochs, trainset, training_loader, valset, validation_loader, loss_fn, optimizer)
    plot_losses(train_loss, val_loss)
    
    # Test
    test_loader = DataLoader(testset, batch_size=32, shuffle=True)
    test_acc, test_loss, _ = evaluate_model(test_loader, testset, model, loss_fn)
    print('Test accuracy: ', test_acc)