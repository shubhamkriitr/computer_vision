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

NUM_WORKERS = 4
try:
    NUM_WORKERS = multiprocessing.cpu_count()
except Exception as exc:
    print("Could not get cpu count")
    print(f"Using {NUM_WORKERS} workers")

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

def distance_batch(x, X):
    N = multiprocessing.cpu_count()
    raise NotImplementedError('distance_batch function not implemented!')

def gaussian(dist, bandwidth):
    return torch.exp(-torch.square(dist)/(2*bandwidth*bandwidth))

def update_point(weight, X):
    return (weight*X.T).T.sum(dim=0)/weight.sum()

def update_point_batch(weight, X):
    raise NotImplementedError('update_point_batch function not implemented!')

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    raise NotImplementedError('meanshift_step_batch function not implemented!')

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        X = meanshift_step(X)   # slow implementation
        # X = meanshift_step_batch(X)   # fast implementation
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
