import numpy as np

def resample_old(particles, particles_w):
    num_samples = particles.shape[0]
    
    sampled_indices = np.random.multinomial(n=1, pvals=particles_w[:, 0], size=num_samples)
    sampled_indices = np.argmax(sampled_indices, axis= 1)
    
    sampled_particles = particles[sampled_indices]
    sampled_particels_w = particles_w[sampled_indices]

    sampled_particels_w = sampled_particels_w/np.sum(sampled_particels_w)

    return sampled_particles, sampled_particels_w

def resample(particles, particles_w):
    num_samples = particles.shape[0]
    
    
    
    sampled_particles = []
    sampled_particels_w = []

    prob_cumsum = 0.0
    prob_bin_size = 1.0/num_samples
    cutoff_threshold = 0.0
    initial_cutoff_threshold = prob_bin_size*np.random.rand()
    
    sample_idx = -1

    sampled_indices = []

    for idx in range(num_samples):
        cutoff_threshold = initial_cutoff_threshold + prob_bin_size*idx
        

        while prob_cumsum < cutoff_threshold:
            sample_idx += 1
            prob_cumsum += particles_w[sample_idx, 0]
        
        sampled_indices.append(sample_idx)

    

    sampled_particles = particles[sampled_indices]
    sampled_particels_w = particles_w[sampled_indices]


    sampled_particels_w = sampled_particels_w/np.sum(sampled_particels_w)

    return sampled_particles, sampled_particels_w