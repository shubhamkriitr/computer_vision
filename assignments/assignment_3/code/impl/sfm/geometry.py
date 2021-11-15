import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # TODO
  # Normalize coordinates (to points on the normalized image plane)

  # These are the keypoints on the normalized image plane (not to be confused with the normalization in the calibration exercise)
  # 
  # FIXME should im1 be normalized before normalizing using K inv

  K_inv = np.linalg.inv(K)
  IM_KP_1 = np.ones((im1.kps.shape[0], 3))
  IM_KP_2 = np.ones((im2.kps.shape[0], 3))
  IM_KP_1[:, 0:2] = im1.kps
  IM_KP_2[:, 0:2] = im2.kps

  normalized_kps2 = np.transpose(K_inv @ np.transpose(IM_KP_2))
  normalized_kps1 = np.transpose(K_inv @ np.transpose(IM_KP_1))

  # TODO
  # Assemble constraint matrix
  constraint_matrix = np.zeros((matches.shape[0], 9))

  for i in range(matches.shape[0]):
    # TODO
    # Add the constraints
    # NOTE kp1' @ E kp2 = 0 for this choice of x and x_dash
    x_im = normalized_kps2[matches[i][1]]
    x_im_dash = normalized_kps1[matches[i][0]]

    x_im = x_im/x_im[2]
    x_im_dash = x_im_dash/x_im_dash[2]

    x = x_im[0]
    y = x_im[1]
    x_dash = x_im_dash[0]
    y_dash = x_im_dash[1]

    constraint_matrix[i, 0] = x_dash*x
    constraint_matrix[i, 1] = x_dash*y
    constraint_matrix[i, 2] = x_dash
    constraint_matrix[i, 3] = y_dash*x
    constraint_matrix[i, 4] = y_dash*y
    constraint_matrix[i, 5] = y_dash
    constraint_matrix[i, 6] = x
    constraint_matrix[i, 7] = y
    constraint_matrix[i, 8] = 1


    pass  # ADD CODE HERE FIXME

  
  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]

  # TODO
  # Reshape the vectorized matrix to it's proper shape again
  E_hat = vectorized_E_hat.reshape(3, 3)

  # TODO
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  U, D_hat, Vh = np.linalg.svd(E_hat)

  #choose new D
  # Since E is up to scale, you can just set the first singular values to 1
  avg_val = (D_hat[0] + D_hat[1])/2.0 
  D = np.diag(np.array([avg_val, avg_val, 0], dtype=E_hat.dtype))
  
  E = U @ D @ Vh

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i,0],:]
    kp2 = normalized_kps2[matches[i,1],:]

    assert(abs(kp1.transpose() @ E @ kp2) < 0.01)

  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`
  # FIXME NOTE add optimized version later
  filtered_points3D = []
  filtered_im1_corrs = []
  filtered_im2_corrs = []
  M_1 = P1[:, 0:3] # M = KR
  M_2 = P2[:, 0:3] # M = KR
  sign_det_M_1 = np.sign(np.linalg.det(M_1))
  sign_det_M_2 = np.sign(np.linalg.det(M_2))
  for i in range(num_new_matches):
    homogeneous_point = np.append(points3D[i], 1)
    homogeneous_point = np.reshape(homogeneous_point, (4,1))
    x_cam_1 = P1 @ homogeneous_point
    is_valid = True

    if sign_det_M_1*x_cam_1[2] < 0:
      is_valid = False

    if is_valid:
      # check 2nd camera
      x_cam_2 = P2 @ homogeneous_point
      if sign_det_M_2*x_cam_2[2] < 0:
        is_valid = False
    
    if not is_valid:
      continue

    filtered_points3D.append(points3D[i])
    filtered_im1_corrs.append(im1_corrs[i])
    filtered_im2_corrs.append(im2_corrs[i])


  points3D = np.array(filtered_points3D, dtype=points3D.dtype)
  im1_corrs = np.array(filtered_im1_corrs, dtype=im1_corrs.dtype)
  im2_corrs = np.array(filtered_im2_corrs, dtype=im2_corrs.dtype)

  return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # TODO
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.
  normalized_points2D = np.transpose(np.linalg.inv(K) @ np.transpose(points2D))

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):

  # TODO 
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

  image = images[image_name]
  points3D = np.zeros((0,3))
  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
  corrs = {}



  return points3D, corrs
  
