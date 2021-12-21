import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib import patches

ARRAY_DTYPE_FLOAT = np.float32
ARRAY_DTYPE_UNIT8 = np.uint8
ARRAY_DTYPE_INT = np.int32

MODEL_NO_MOTION = 0
MODEL_CONST_VELOCITY = 1

A_no_motion = [[1, 0],
               [0, 1]]
A_no_motion = np.array(A_no_motion, dtype=ARRAY_DTYPE_FLOAT)
A_const_velocity = [[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]
A_const_velocity = np.array(A_const_velocity, dtype=ARRAY_DTYPE_FLOAT)


def propagate(particles, frame_height, frame_width, params):
    args = particles, frame_height, frame_width, params
    model_code = params["model"]
    if model_code == MODEL_NO_MOTION:
        return propagate_no_motion(*args)
    elif model_code == MODEL_CONST_VELOCITY:
        return propagate_constant_velocity(*args)
    else:
        print("Warning: Unknown model code {}".format(model_code))
        return None


def propagate_no_motion(particles, frame_height, frame_width, params):
    A = A_no_motion
    sigma_position = params['sigma_position']

    w = np.random.normal(loc=0.0, scale=sigma_position, size=particles.shape)

    new_particles = (A@particles.T).T + w
    new_particles = clip_values(new_particles, frame_height, frame_width)
    return new_particles

def propagate_constant_velocity(particles, frame_height, frame_width, params):
    A = A_const_velocity
    sigma_position = params['sigma_position']
    sigma_velocity = params['sigma_velocity']

    w_pos = np.random.normal(loc=0.0, scale=sigma_position, size=(particles.shape[0], 2))
    w_vel = np.random.normal(loc=0.0, scale=sigma_velocity, size=(particles.shape[0], 2))
    w = np.concatenate([w_pos, w_vel], axis=1)

    new_particles = (A@particles.T).T + w
    new_particles = clip_values(new_particles, frame_height, frame_width)
    return new_particles

def clip_values(particles, frame_height, frame_width):
    x_min = 0.0
    y_min = 0.0
    x_max = frame_width
    y_max = frame_height

    x_idx = 1
    y_idx = 0
    
    particles[:, x_idx] = np.clip(particles[:, x_idx], a_min=x_min, a_max=x_max)

    particles[:, y_idx] = np.clip(particles[:, y_idx], a_min=y_min, a_max=y_max)

    return particles

