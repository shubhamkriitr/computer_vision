import numpy as np
from color_histogram import color_histogram

def observe(particles, frame, bbox_height, bbox_width, hist_bin, target_hist, sigma_observe):
    
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    particles_hist = np.zeros(particles.shape[0], hist_bin*3) # 3 channels

    for idx in range(particles.shape[0]):
        x_coord = particles[idx, 1]
        y_coord = particles[idx, 0] # NOTE: For particles first dim is Y 
        # but for mean_state_a_posteriori first dim is X
        particles_hist[idx, :] = color_histogram(
                                      min(max(0, round(x_coord-0.5*bbox_width)), frame_width-1),
                                      min(max(0, round(y_coord-0.5*bbox_height)), frame_height-1),
                                      min(max(0, round(x_coord+0.5*bbox_width)), frame_width-1),
                                      min(max(0, round(y_coord+0.5*bbox_height)), frame_height-1),
                                      frame, hist_bin)
    
    # CALL: compute_particle_weights
    

def compute_particle_weights(particles_hist, target_hist, sigma_observe):
    pass