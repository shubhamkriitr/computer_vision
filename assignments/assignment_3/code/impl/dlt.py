import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))

  for i in range(num_corrs):
    # TODO Add your code here
    A = compute_one_constraint_row(points2D[i], points3D[i])
    constraint_matrix[i*2:i*2+2] = A


  return constraint_matrix


def compute_one_constraint_row(point2D, point3D):
  # point 2D (x, y, 1) Point3D (X, Y, Z, 1)
  Xt = np.append(point3D, 1)
  x = point2D[0]
  y = point2D[1]
  A = np.zeros((2, 12), dtype=point2D.dtype)

  A[0, 4:8] = -Xt # -w*Xt
  A[0, 8:12] = y*Xt
  A[1, 0:4] = Xt
  A[1, 8:12] = -x*Xt

  return A