import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

# for batch ops
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from torch.autograd import backward
torch.manual_seed(0)
np.random.seed(0)


def distance(x, X):
    return (X-x).norm(p=2, dim=1)

def distance_batch_(x, X):
    # x = n x d ; X = N x d -> n x N
    return torch.cdist(x, X)

def gaussian(dist, bandwidth):
    return torch.exp(-torch.square(dist)/(2*bandwidth*bandwidth))

def update_point(weight, X):
    return (weight*X.T).T.sum(dim=0)/weight.sum()

def update_point_batch_(weight, X):
    wt_sum = weight.sum(axis=1)
    weighted_sum = torch.matmul(weight, X)
    return weighted_sum/torch.unsqueeze(wt_sum, dim=1)


def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_


def meanshift_step_batch_(X, bandwidth=2.5):
    dist = distance_batch_(X, X)
    wt = gaussian(dist, bandwidth)
    X_ = update_point_batch_(wt, X)
    return X_




def meanshift(X):
    X = X.clone()
    for _ in range(20):
        X = meanshift_step(X)   # slow implementation
        # X = meanshift_step_batch(X)   # fast implementation
        # X = meanshift_step_batch_(X)   # fast implementation
        
        # 
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)

