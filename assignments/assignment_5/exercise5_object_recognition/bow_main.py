import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm

NUM_CELLS = (4, 4)  # from handout: section 2.1.2
HALF_NUM_CELLS_H = NUM_CELLS[0]//2
HALF_NUM_CELLS_W = NUM_CELLS[1]//2

DATA_TYPE_INDEX = np.int32
PI_RAD = 3.14159265
THETA_MAX = 2*PI_RAD
TOLERANCE_NEAR_ZERO = THETA_MAX/(8*2) # NBins = 8 k =2
SCALED_ANGLE_RANGE = (0, 2*PI_RAD) # it will be used to create bins


def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    vPoints = None  # numpy array, [nPointsX*nPointsY, 2]

    # todo:A
    if len(img.shape) > 2:
        print("warning: Image array dim > 2")
    h, w = img.shape[0], img.shape[1]

    # Assuming w->x / h->y
    w_offset = border
    h_offset = border
    w_end = w - border # exclusive
    h_end = h - border

    num_w_pts_within_border = w - 2*border
    num_h_pts_within_border = h - 2*border

    if num_h_pts_within_border < nPointsY:
        print("warning: num_h_pts_within_border({}) < nPointsY({})".format(
            num_h_pts_within_border, nPointsY
        ))
    if num_w_pts_within_border < nPointsX:
        print("warning: num_w_pts_within_border({}) < nPointsX({})".format(
            num_w_pts_within_border, nPointsX
        ))

    w_grid_step_size = num_w_pts_within_border/nPointsX
    h_grid_step_size = num_h_pts_within_border/nPointsY

    w_abscissa = np.arange(w_offset+w_grid_step_size/2.0, w_end + 1, w_grid_step_size)
    h_ordinate = np.arange(h_offset+h_grid_step_size/2.0, h_end + 1, h_grid_step_size)

    w_abscissa = w_abscissa[0:nPointsX]
    h_ordinate = h_ordinate[0:nPointsY]

    w_abscissa = np.rint(w_abscissa)
    h_ordinate = np.rint(h_ordinate)

    w_abscissa = w_abscissa.astype(DATA_TYPE_INDEX)
    h_ordinate = h_ordinate.astype(DATA_TYPE_INDEX)

    y, x = np.meshgrid(h_ordinate, w_abscissa)

    vPoints = np.vstack([y.ravel(), x.ravel()]).T

    return vPoints



def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    nBins = 8
    w = cellWidth
    h = cellHeight

    grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1)

    grad_x = grad_x.astype(np.float32)
    grad_y = grad_y.astype(np.float32)

    # grad_magnitude = np.sqrt(grad_x*grad_x + grad_y*grad_y)
    # grad_drn = np.arctan(grad_y/grad_x)

    grad_magnitude, grad_drn = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=False)
    grad_drn = _process_angle(grad_drn)

    w_pixel_offset = HALF_NUM_CELLS_W*cellWidth
    h_pixel_offset = HALF_NUM_CELLS_H*cellHeight

    base_mesh_grid = _create_base_mesh_grid(cellHeight, cellWidth)

    descriptors = []  # list of descriptors for the current image, each entry is one 128-d vector for a grid point
    for i in range(len(vPoints)):
        # todo:A?
        h_center = vPoints[i][0]
        w_center = vPoints[i][1]

        h_base = h_center - h_pixel_offset
        w_base = w_center - w_pixel_offset

        current_descriptor = _get_descriptors_hog_for_one_patch_group(
            grad_magnitude, grad_drn, nBins, cellWidth, cellHeight, base_mesh_grid, w_base, h_base)
        
        descriptors.append(current_descriptor)


    descriptors = np.asarray(descriptors) # [nPointsX*nPointsY, 128], descriptor for the current image (100 grid points)
    return descriptors

def _process_angle_old(grad_drn):

    # bring in 0 to pi range
    grad_drn[grad_drn > PI_RAD] =  grad_drn[grad_drn > PI_RAD] - PI_RAD

    condn = np.where(grad_drn > (PI_RAD - TOLERANCE_NEAR_ZERO))
    grad_drn[condn] = 0

    grad_drn = grad_drn/PI_RAD
    grad_drn = np.clip(grad_drn, SCALED_ANGLE_RANGE[0], SCALED_ANGLE_RANGE[1])

    return grad_drn 

def _process_angle(grad_drn):

    
    grad_drn = grad_drn - TOLERANCE_NEAR_ZERO/2.0
    condn = np.where(grad_drn < 0)
    grad_drn[condn] = 2*PI_RAD + grad_drn[condn]

    # grad_drn = grad_drn/(2*PI_RAD)
    grad_drn = np.clip(grad_drn, SCALED_ANGLE_RANGE[0], SCALED_ANGLE_RANGE[1])

    return grad_drn 

def _process_angle_3(grad_drn):

    
    grad_drn = np.clip(grad_drn, SCALED_ANGLE_RANGE[0], SCALED_ANGLE_RANGE[1])

    return grad_drn 

def _create_base_mesh_grid(num_pixels_h, num_pixels_w):
    w_abscissa = np.arange(0, num_pixels_w, 1, dtype=np.int16)
    h_ordinate = np.arange(0, num_pixels_h, 1, dtype=np.int16)
    

    y, x = np.meshgrid(h_ordinate, w_abscissa)

    mesh_grid = np.vstack([y.ravel(), x.ravel()]).T
    
    return mesh_grid

def _get_descriptors_hog_for_one_patch_group(grad_magnitude, grad_drn, nBins, cellWidth, cellHeight, base_mesh_grid, w_base, h_base):
    features = []

    for h_idx in range(NUM_CELLS[0]):
        for w_idx in range(NUM_CELLS[1]):

            w_begin = w_base + w_idx * cellWidth
            h_begin = h_base + h_idx * cellHeight

            current_grid = base_mesh_grid + np.array([h_begin, w_begin])

            grad_magnitude_slice = grad_magnitude[current_grid[:,0], current_grid[:,1]]
            grad_drn_slice = grad_drn[current_grid[:,0], current_grid[:,1]]

            curr_feature, bins_created = np.histogram(
                grad_drn_slice, bins=nBins, range=SCALED_ANGLE_RANGE, normed=None,
                 weights=grad_magnitude_slice, density=None)
            
            features.append(curr_feature)
    
    features = np.hstack(features)

    return features


def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    vFeatures = []  # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    # Extract features for all image
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # Collect local feature points for each image, and compute a descriptor for each local feature point
        # todo:A?
        grid_vPoints = grid_points(img, nPointsX, nPointsY, border)
        current_image_features = descriptors_hog(img, grid_vPoints, cellWidth, cellHeight)

        vFeatures.append(current_image_features)


    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])  # [n_imgs*n_vPoints, 128]
    print('number of extracted features: ', len(vFeatures))


    # Cluster the features using K-Means
    print('clustering ...')
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter).fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_  # [k, 128]
    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    histo = None

    # todo:A
    dist_matrix = []
    num_centers = vCenters.shape[0]
    for idx in range(num_centers):
        diff = vFeatures - vCenters[idx]
        distance = np.linalg.norm(diff, ord=2, axis=1)
        dist_matrix.append(distance)
    
    dist_matrix = np.vstack(dist_matrix)
    dist_argmin = np.argmin(dist_matrix, axis=0)
    # print(dist_matrix)
    # print(dist_argmin)
    histo, bins_created = np.histogram(dist_argmin, bins=num_centers, range=(0, num_centers), normed=None,
                 weights=None, density=None)
        
    return histo





def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    vBoW = []
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # todo:A
        grid_vPoints = grid_points(img, nPointsX, nPointsY, border)
        current_image_features = descriptors_hog(img, grid_vPoints, cellWidth, cellHeight)
        current_bow_histogram = bow_histogram(current_image_features, vCenters)
        vBoW.append(current_bow_histogram)


    vBoW = np.asarray(vBoW)  # [n_imgs, k]
    return vBoW



def bow_recognition_nearest(histogram,vBoWPos,vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    DistPos, DistNeg = None, None

    # Find the nearest neighbor in the positive and negative sets and decide based on this neighbor
    # todo

    DistPos = np.min(np.linalg.norm(vBoWPos - histogram, ord=2, axis=1))
    DistNeg = np.min(np.linalg.norm(vBoWNeg - histogram, ord=2, axis=1))
    

    if (DistPos < DistNeg):
        sLabel = 1
    else:
        sLabel = 0
    return sLabel





if __name__ == '__main__':
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'

    # fix seed for reproducibility
    np.random.seed(0)
    cv2.setRNGSeed(0)
    
    import sys
    k = 6  # todo:A
    numiter = 40  # todo:A
    print("="*80)
    print("Using k={} and numiter={}".format(k, numiter))

    print('creating codebook ...')
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

    print('creating bow histograms (pos) ...')
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print('creating bow histograms (neg) ...')
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print('creating bow histograms for test set (pos) ...')
    vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print('testing pos samples ...')
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWPos_test[i:(i+1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print('test pos sample accuracy:', acc_pos)

    # test neg samples
    print('creating bow histograms for test set (neg) ...')
    vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print('testing neg samples ...')
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print('test neg sample accuracy:', acc_neg)
