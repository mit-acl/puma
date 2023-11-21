#!/usr/bin/env python3

" SGC model using torch geometric with Cora"

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SGConv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F

" ********************* IMPORT DATA ********************* "

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0] # Tensor representation of the Cora-Planetoid data
print("Coda: ", data)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

" ********************* CONSTRUCT THE MODEL ********************* "

SGC_model = SGConv(in_channels=data.num_features, # number of features
                   out_channels=dataset.num_classes, # dimension of embedding
                   K = 1, cached = True)

" ********************* GET EMBEDDING ********************* "
print(" shape of the original data: ", data.x.shape)
print("shape of the embedding data: ", SGC_model(data.x, data.edge_index).shape)

" ********************* CONSTRUCT THE MODEL FOR CLASSIFICATION ********************* "
class SGCNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SGConv(in_channels=data.num_features, # number of features
                            out_channels=dataset.num_classes,
                            K =1,
                            cached=True)
        
    def forward(self):
        x = self.conv1(data.x, data.edge_index)

        # computation of log fosftmax
        return F.log_softmax(x, dim=1)

SGC_model, data = SGCNet().to(device), data.to(device)
optimizer = torch.optim.Adam(SGC_model.parameters(), lr=0.2, weight_decay=5e-4)

# What are the learing parameters:
for i, parameter in SGC_model.named_parameters():
    print(" Parameter {}".format(i))
    print("Shape: ", parameter.shape)

" ********************* TRAIN FUNCTION ********************* "

def train():
    SGC_model.train() # set the model.training to be True
    optimizer.zero_grad() # reset the gradient of all variables
    predicted_y = SGC_model() # predicted y in log softmax prob
    true_y = data.y # true y
    losses = F.nll_loss(predicted_y[data.train_mask], true_y[data.train_mask])
    losses.backward()
    optimizer.step() # update the parameters such that the loss is minimized
    
" ********************* TEST FUNCTION ********************* "

def test():
    SGC_model.eval() # set the model.training to be False
    logits = SGC_model() # Log prob of all data
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1] # get the index of the max log prob
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

" ********************* TRAINING AND TESTING ********************* "
best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))