import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import model_selection
# Growing Spheres:
from scipy.special import gammainc
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from torch.utils.data import Dataset, DataLoader

# writer = SummaryWriter()
BATCH_SIZE = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data, labels = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.2, random_state=None)
#plt.scatter(data.T[0],data.T[1])

# Divide training test
data_train, data_test, labels_train, labels_test =  model_selection.train_test_split(data, labels, test_size=.5, train_size=.5, random_state=None, shuffle=True, stratify=None)
#plt.scatter(data_train.T[0], data_train.T[1], c = labels_train)

class MyDataset(Dataset):
    def __init__(self, data, labels) -> None:
        super().__init__()
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.float32)
    
    def __getitem__(self, index):
        return (self.data[index], self.labels[index])
    
    def __len__(self):
        return self.data.shape[0]

data_train = DataLoader ( MyDataset(data_train,labels_train) , shuffle=True , batch_size=BATCH_SIZE)


class BinaryClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim) 
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_out = nn.Linear(hidden_dim, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.layers = nn.Sequential(self.layer_1,
                                    self.relu,
                                    self.batchnorm1,
                                    self.layer_2,
                                    self.relu,
                                    self.batchnorm2,
                                    self.dropout,
                                    self.layer_out,
                                    )
        
    def forward(self, inputs):
        
        return self.layers(inputs)    

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

NB_ITERATIONS = 100

model = BinaryClassification(2, 16)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters())

for epoch in tqdm(range(NB_ITERATIONS)):
    for x,y in data_train:
        optim.zero_grad()
        x = x.to(device)
        y = y.to(device)
        yhat = model(x)
        acc = binary_acc(yhat, y.unsqueeze(1))
        #print("y", y)
        #print("yhat", yhat)
        print("acc", acc)
        loss = criterion(yhat, y.unsqueeze(1))
        
        loss.backward()
        optim.step()
        print("loss", loss )