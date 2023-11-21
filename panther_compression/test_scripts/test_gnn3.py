#!/usr/bin/env python3

" Workshop on DeepWalk Algorithm using Karate Club "

import networkx as nx
import matplotlib.pyplot as plt
from karateclub import DeepWalk
from karateclub import Node2Vec
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

G = nx.karate_club_graph()
print("Number of nodes: ", G.number_of_nodes())
# nx.draw_networkx(G)
# plt.show()

" plot th egraph with labels "
labels = []
for i in G.nodes:
    club_names = G.nodes[i]['club']
    labels.append(1 if club_names == 'Officer' else 0)

layout_pos = nx.spring_layout(G)
# nx.draw_networkx(G, pos=layout_pos, node_color=labels, cmap='coolwarm')
# plt.show()

" Perform node embedding using DeepWalk Algorithm "
Deepwalk_model = DeepWalk(walk_number=10, walk_length=80, dimensions=124)
Deepwalk_model.fit(G)
embedding = Deepwalk_model.get_embedding()
print("Embedding shape: ", embedding.shape)

" Perform node embedding using Node2Vec Algorithm "
N2vec_model = Node2Vec(walk_number=10, walk_length=80, p=0.6, q=0.4, dimensions=124)
N2vec_model.fit(G)
N2vec_embedding = N2vec_model.get_embedding()
print("Embedding shape: ", N2vec_embedding.shape)

" Low dimensional plot of the nodes x features "
PCA_model = sklearn.decomposition.PCA(n_components=2)
lowdimension_emdedding = PCA_model.fit_transform(embedding)
print("Low dimensional embeding represenaion shape: ", lowdimension_emdedding.shape)
plt.scatter(lowdimension_emdedding[:, 0], lowdimension_emdedding[:, 1], c=labels, s=15, cmap='coolwarm')
plt.show()

N2vec_lowdimension_emdedding = PCA_model.fit_transform(N2vec_embedding)
print("Low dimensional embeding represenaion shape: ", N2vec_lowdimension_emdedding.shape)
plt.scatter(N2vec_lowdimension_emdedding[:, 0], N2vec_lowdimension_emdedding[:, 1], c=labels, s=15, cmap='coolwarm')
plt.show()

" Node classification using embedded model "
x_train, x_test, y_train, y_test = train_test_split(embedding, labels, test_size=0.3)
ML_model = LogisticRegression(random_state=0).fit(x_train, y_train)
y_pred = ML_model.predict(x_test)
ML_acc = roc_auc_score(y_test, y_pred)
print("Accuracy of the model: ", ML_acc)

nx_train, nx_test, ny_train, ny_test = train_test_split(N2vec_embedding, labels, test_size=0.3)
nML_model = LogisticRegression(random_state=0).fit(nx_train, ny_train)
ny_pred = nML_model.predict(nx_test)
nML_acc = roc_auc_score(ny_test, ny_pred)
print("Nede2vec Accuracy of the model: ", nML_acc)