import numpy as np
import scipy.optimize as spo

from impl.util import MakeHomogeneous, HNormalize

# Compute the reprojection error for a single correspondence
def ReprojectionError(P, point3D, point2D):
    # TODO
    # Project the 3D point into the image and compare it to the keypoint.
    # Make sure to properly normalize homogeneous coordinates.
    X = np.append(point3D, 1)
    x = np.append(point2D, 1)

    x_dash = P @ X
    x_dash = x_dash/x_dash[2]
    # d_wi = np.linalg.norm(x - x_dash) - only diff needs to be sent
    # as norm and squaring is done in OptimizeProjectionMatrix function
    err = x - x_dash

    err = err[0:2] # remove last dim
    return err

# Compute the residuals for all correspondences of the image
def ImageResiduals(P, points2D, points3D):

  num_residuals = points2D.shape[0]
  res = np.zeros(num_residuals*2)

  for res_idx in range(num_residuals):
    p3D = points3D[res_idx]
    p2D = points2D[res_idx]

    err = ReprojectionError(P, p3D, p2D)

    res[res_idx*2:res_idx*2+2] = err
  
  return res

# Optimize the projection matrix given the 2D-3D point correspondences.
# 2D and 3D points with the same index are assumed to correspond.
def OptimizeProjectionMatrix(P, points2D, points3D):

    # The optimization requires a scalar cost value.
    # We use the sum of squared differences of all correspondences
    f = lambda x : np.linalg.norm(ImageResiduals(np.reshape(x, (3, 4)), points2D, points3D)) ** 2

    # Since the projection matrix is scale invariant we have an open degree of freedom from just the constraints.
    # Make sure this is fixed by keeping the last component close to 1.
    scale_constraint = {'type': 'eq', 'fun': lambda x : x[11] - 1}

    iter_count = 0
    def callback_to_display_err(*args, **kwargs):
      nonlocal iter_count
      iter_count += 1
      x = args[0]
      err = np.linalg.norm(ImageResiduals(np.reshape(x, (3, 4)), points2D, points3D)) ** 2
      print(f"Iteration: {iter_count} Error: {err}")

    # Make sure the scale constraint is fulfilled at the beginning
    result = spo.minimize(f, np.reshape(P / P[2,3], 12), options={'disp': True}, constraints=[scale_constraint], tol=1e-12,
                           callback=callback_to_display_err)
    return np.reshape(result.x, (3, 4))