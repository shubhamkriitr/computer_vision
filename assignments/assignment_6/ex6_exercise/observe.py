import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost

def observe(particles, frame, bbox_height, bbox_width, hist_bin, target_hist, sigma_observe):
    
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    particles_hist = np.zeros((particles.shape[0], hist_bin*3)) # 3 channels

    for idx in range(particles.shape[0]):
        x_coord = particles[idx, 0]
        y_coord = particles[idx, 1] # NOTE: For particles and  mean_state_a_posteriori first dim is X
        particles_hist[idx, :] = color_histogram(
                                      min(max(0, round(x_coord-0.5*bbox_width)), frame_width-1),
                                      min(max(0, round(y_coord-0.5*bbox_height)), frame_height-1),
                                      min(max(0, round(x_coord+0.5*bbox_width)), frame_width-1),
                                      min(max(0, round(y_coord+0.5*bbox_height)), frame_height-1),
                                      frame, hist_bin)
    
    # CALL: compute_particle_weights
    particles_w = compute_particle_weights(particles_hist, target_hist, sigma_observe)
    return particles_w
    

def compute_particle_weights(particles_hist, target_hist, sigma_observe):
    chi_sqr_dist = np.zeros((particles_hist.shape[0], 1))
    for idx in range(particles_hist.shape[0]):
        chi_sqr_dist[idx, 0] = chi2_cost(particles_hist[idx], target_hist)
    dnr = np.sqrt(2*np.pi)*sigma_observe
    particles_w = np.exp(-(chi_sqr_dist*chi_sqr_dist)/(2*sigma_observe*sigma_observe))/dnr

    if np.sum(particles_w) < 1e-5:
        particles_w = np.ones_like(particles_w)

    particles_w = particles_w/np.sum(particles_w) # Normalize this PD

    return particles_w