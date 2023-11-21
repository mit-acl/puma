#!/usr/bin/env python3

" Node2Vec model using torch geometric with Cora "
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

" ********************* IMPORT DATA ********************* "

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0] # Tensor representation of the Cora-Planetoid data
print("Coda: ", data)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

" ********************* CONSTRUCT THE MODEL ********************* "

Node2Vec_model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True).to(device)
loader = Node2Vec_model.loader(batch_size=128, shuffle=True, num_workers=4)
optimizer = torch.optim.SparseAdam(list(Node2Vec_model.parameters()), lr=0.01)


" ********************* TRAIN FUNCTION ********************* "

def train():
    Node2Vec_model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader: # positive and negative random walks
        optimizer.zero_grad() # reset of the gradient of all variables
        loss = Node2Vec_model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

" ********************* GET EMBEDDING ********************* "

for epoch in range(1, 101):
    loss = train()
    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))

" ********************* PLOT 2D OF EMBEDDED REP ********************* "
@torch.no_grad() # Deactivate autograd functionality
def plot_point(colors):
    Node2Vec_model.eval() # Evaluate the model based on the trained parameters
    z = Node2Vec_model(torch.arange(data.num_nodes).to(device)) # Embedding rep
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()
    plt.figure()
    for i in range(dataset.num_classes):
        plt.scatter(z[y==i, 0], z[y==i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.show()

colors = ['#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#ff0000', '#0000ff']

plot_point(colors)

" ********************* NODE CLASSIFICATION ********************* "

def test():
    Node2Vec_model.eval()
    z = Node2Vec_model(torch.arange(data.num_nodes).to(device))
    acc = Node2Vec_model.test(z[data.train_mask], data.y[data.train_mask], z[data.test_mask], data.y[data.test_mask], max_iter=150)
    return acc

print('final accuracy: ', test())