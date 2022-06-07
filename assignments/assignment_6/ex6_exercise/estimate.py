import numpy as np

def estimate(particles, particles_weight):
    weighted_particles = particles_weight*particles
    expected_particle = np.sum(weighted_particles, axis=0)/np.sum(particles_weight)
    return expected_particle