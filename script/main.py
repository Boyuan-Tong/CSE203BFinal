# Standard imports
import torch
import torch.optim as optim
import numpy as np
import scipy.io
from helper import *
import struct
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Loading MNIST dataset

with open('../data/train-images.idx3-ubyte', 'rb') as f:
    magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
    TrainImgs = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)

with open('../data/train-labels.idx1-ubyte', 'rb') as f:
    magic, num = struct.unpack('>II', f.read(8))
    TrainLabs = np.fromfile(f, dtype=np.uint8)

with open('../data/t10k-images.idx3-ubyte', 'rb') as f:
    magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
    TestImgs = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)

with open('../data/t10k-labels.idx1-ubyte', 'rb') as f:
    magic, num = struct.unpack('>II', f.read(8))
    TestLabs = np.fromfile(f, dtype=np.uint8)

print("Data Load Finished")

# Loading the confusion matrix
confusionMatrix = loadConfusionMatrix()
confusionMatrix = torch.tensor(confusionMatrix.astype(float))
distanceMatrix = 1 - confusionMatrix
print(distanceMatrix)

# Calculating the feature vector matrix according to average of images
rowFeatureVectors = []
imgs = [[] for i in range(10)]
for i in range(TrainImgs.shape[0]):
    imgs[TrainLabs[i]].append(TrainImgs[i])

for i in range(10):
    rowFeatureVectors.append(np.mean(np.array(imgs[i]), axis=0))
x= rowFeatureVectors[5]
pixels = x.reshape((28,28))/256
plt.imshow(pixels,cmap="gray")
plt.show()

P = []
for i in range(10):
    for j in range(10):
        P.append(np.abs(rowFeatureVectors[i] - rowFeatureVectors[j]))

rowFeatureVectors = torch.tensor(P) # 100 * 784

# Initializing the distance metric
M = torch.rand(784, 784)

# Setting the number of epochs
num_epochs = 2
M.requires_grad = True

# Setting the ground truth variable as the confusion matrix
groundTruth = distanceMatrix

# Setting different optimizer to be used to optimize the parameters of the distance matrix
optimizer_adam = optim.Adam([M], lr=0.0001, weight_decay=0.00005)
optimizer_sgd = optim.SGD([M], lr=0.0001)

# Defining variables to store the logging values of during optimizing
best_loss = 100000
best_epoch = 0
best_estimate = 0

for ep in range(num_epochs):
    output = torch.mm(torch.mm(rowFeatureVectors, M.double()), rowFeatureVectors.T)
    output = torch.reshape(torch.diag(output), (10, 10))
    loss = MSELoss(output, groundTruth)
    loss.backward()
    optimizer_sgd.step()
    if (loss.item() < best_loss):
        best_loss = loss.item()
        best_estimate = output
        best_epoch = ep
    print("The loss for ", ep, "iteration is: ", loss.item())

with open('../data/W.npy', 'wb') as f:
    np.save(f, M.detach().numpy())

print("The best loss is at ", best_epoch, "iteration and the best loss is: ", best_loss)

print("The matrix which indicates the difference between the distance matrix and estimated distance matrix is:",
      (best_estimate - distanceMatrix).detach().numpy())

# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(TrainImgs, TrainLabs)
# print(knn.score(TestImgs, TestLabs))

dist = lambda x, y: np.transpose(x-y) @ M.detach().numpy() @ (x-y)
knn2 = KNeighborsClassifier(n_neighbors=1, metric=dist)
knn2.fit(TrainImgs[:2000], TrainLabs[:2000])
print(knn2.score(TestImgs[:100], TestLabs[:100]))

