import torch
import torch.nn
import numpy as np


# Z scoring the matrix
def normalize(matrix):
    mean = torch.mean(matrix.flatten())
    sigma = torch.std(matrix.flatten())
    matrix = (matrix - mean) / sigma
    return matrix


# Making a non-symmetrix square matrix symmetric
def makeSymmetric(matrix):
    return 0.5 * (matrix + matrix.T)


# Defining some loss functions here

def MSELoss(mat1, mat2):
    loss = torch.nn.MSELoss()
    return loss(mat1, mat2)


def loadConfusionMatrix():
    M = [[954, 0, 0, 7, 1, 10, 6, 3, 7, 3],
         [0, 1031, 4, 3, 1, 4, 1, 2, 16, 2],
         [12, 21, 852, 18, 11, 8, 14, 20, 29, 5],
         [2, 5, 9, 899, 1, 71, 0, 12, 23, 7],
         [2, 8, 2, 2, 861, 7, 7, 1, 4, 89],
         [7, 5, 9, 24, 3, 833, 12, 8, 12, 2],
         [11, 6, 2, 0, 6, 31, 902, 0, 8, 1],
         [3, 10, 5, 3, 7, 7, 1, 1041, 0, 14],
         [2, 28, 4, 29, 2, 31, 1, 9, 882, 21],
         [7, 3, 1, 7, 10, 11, 1, 44, 4, 873]]
    M = np.array(M)
    M = np.transpose(np.transpose(M)/np.sum(M, axis=1))
    return M