import numpy as np

def resample(particles, particles_w):
    num_samples = particles.shape[0]
    
    sampled_indices = np.random.multinomial(n=1, pvals=particles_w[:, 0], size=num_samples)
    sampled_indices = np.argmax(sampled_indices, axis= 1)
    
    sampled_particles = particles[sampled_indices]
    sampled_particels_w = particles_w[sampled_indices]

    sampled_particels_w = sampled_particels_w/np.sum(sampled_particels_w)
    sampled_particels_w[-1 ,0] = 1.0 - np.sum(sampled_particels_w[:-1, 0])

    return sampled_particles, sampled_particels_w
