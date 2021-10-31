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
NUM_WORKERS = 8
MAIN_EXECUTOR = ProcessPoolExecutor(max_workers=NUM_WORKERS)
# try:
#     # NUM_WORKERS = multiprocessing.cpu_count()
# except Exception as exc:
#     print("Could not get cpu count")

def get_batch_indices(begin_, end_, workers):
    batches = []
    num_records = end_ - begin_ # end_ is exclusive
    start = begin_
    batch_size = num_records//workers
    spill_over = num_records%workers
    while start < end_:
        this_end = start + batch_size
        if spill_over > 0:
            this_end += 1
            spill_over -= 1
        batches.append([start, this_end])
        start = this_end
    if len(batches) > 0 and batches[-1][1] > end_:
        batches[-1][1] = end_
    return batches 


def distance(x, X):
    return (X-x).norm(p=2, dim=1)


def _distance_batch(args):
    # print(f"dist: {args[0].shape} - {args[1].shape}")
    return distance(args[0], args[1])

def distance_batch(x, X):
    batch_idx = get_batch_indices(0, X.shape[0], NUM_WORKERS)
    args_arr = [(x, X[start_:end_]) for start_, end_ in batch_idx]
    results = []
    executor = MAIN_EXECUTOR
    for sub_result in executor.map(_distance_batch, args_arr):
        results.append(sub_result)
    return torch.cat(results, dim=0)

def distance_batch_(x, X):
    # x = n x d ; X = N x d -> n x N
    return torch.cdist(x, X)

def gaussian(dist, bandwidth):
    return torch.exp(-torch.square(dist)/(2*bandwidth*bandwidth))

def update_point(weight, X):
    return (weight*X.T).T.sum(dim=0)/weight.sum()


def _update_point_nodiv(args):
    # print(f"update: {args[0].shape} - {args[1].shape}")
    return (args[0]*args[1].T).T.sum(dim=0)

def update_point_batch(weight, X):
    wt_sum = weight.sum()
    running_sum = 0
    batch_idx = get_batch_indices(0, X.shape[0], NUM_WORKERS)
    args_arr = [(weight[start_:end_], X[start_:end_]) for start_, end_ in batch_idx]
    results = []
    executor = MAIN_EXECUTOR
    # with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for sub_result in executor.map(_update_point_nodiv, args_arr):
        running_sum += sub_result
    return running_sum/wt_sum

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

def _meanshift_step_batch(args):
    X_part = args[0]
    X = args[1]
    bandwidth = args[2]
    X_ = X_part.clone()
    # for i, x in enumerate(X):
    for i in range(X_part.shape[0]):
        x = X_part[i]
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    batch_idx = get_batch_indices(0, X.shape[0], NUM_WORKERS)
    args_arr = [(X[start_:end_], X, bandwidth) for start_, end_ in batch_idx]
    X_ = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for x_ in executor.map(_meanshift_step_batch, args_arr):
            X_.append(x_)
    
    return torch.cat(X_)

def meanshift_step_batch_(X, bandwidth=2.5):
    dist = distance_batch_(X, X)
    wt = gaussian(dist, bandwidth)
    X_ = update_point_batch_(wt, X)
    return X_




def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        print(f"Using {NUM_WORKERS} workers")
        # X = meanshift_step_batch(X)   # fast implementation
        X = meanshift_step_batch_(X)   # fast implementation
        
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
MAIN_EXECUTOR.shutdown()
