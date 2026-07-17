import numpy as np
import matplotlib.pyplot as plt




from BOWaves.utilities import sikmeans_utils, wrappers
from scipy.optimize import nnls
from scipy.sparse.linalg import svds




# MAIN FUNCTION


# Window lenght of each atom
P = 128


# Square root of dictionary size
K_true_sqrt = 14


# Total number of wavelets in dictionary (196)
K_true = K_true_sqrt**2


# Factor for signal lenght
L = 2*P


# Number of activation steps
# T = 4000*L

# Smaller amount for testing purposes
T = 200*L


# Random value creation
rng = np.random.default_rng(0)


# Creates bounds for the atoms
height_min = 0.01
height_max_lim = [0.05,1]
rate_lim = [-4.5,-2.5]


# This function will create the Gabor Wavelet time series
def generate_time_series_with_dict(P,K_true,rate_lim,height_min,height_max_lim,rng):


    # Sampling frequency
    fs=128


    # Generates (196) random frequencies in the shape (196,1)
    f = 10**rng.uniform(0,np.log10(35),K_true)[:,np.newaxis]


    # Generates random time delays for each wavelet
    t0 = rng.uniform(P/3,2*P/3,K_true)[:,np.newaxis]


    # Generates random decay envelopes in each wavlet
    a =  rng.uniform(P/6,P/3,K_true)[:,np.newaxis]


    # Creates a time vector from 0 to 127
    t = np.arange(P)[np.newaxis,:]


    # Computes the 196 dictionary wavelets to turn them into Gabor Wavelets
    # The Shape of C_true is then (196, 128) - 196 wavelets with 128 samples each
    C_true = np.real(np.exp(-1j*2*np.pi*f/fs*(t-t0)-(t-t0)**2/a**2))


    # Sets maximum highs and ranges for the wavelets
    rate_range = np.logspace(rate_lim[0],rate_lim[1],K_true_sqrt)
    height_range = np.linspace(height_max_lim[0],height_max_lim[1],K_true_sqrt)


    # Cartesian product of heights and frequiences
    R,H = np.meshgrid(rate_range,height_range)


    rates = R.ravel()[np.newaxis,:]
    highs = H.ravel()[np.newaxis,:]


    # Creates Spike Slab for wavelets
    act = (rng.uniform(0,1,(T,K_true)) < rates)*(rng.uniform(height_min*np.ones((T,1)),highs)) # spike-slab


    # Convoles each wavelet with the activation of the spike-train
    # This matrix will have the size (196, 1024127)
    X = np.vstack([np.convolve(C_true[i],act[:,i],mode='full') for i in range(K_true)])
   
    return C_true,rates,highs, np.sum( X,axis=0)
    # This function will return:
    # C_True - The dictionary of the 196 Gabor Wavlets in the size (196,128)
    # rates - Probability of each Gabor Wavelet triggering at any moment
    # highs - The upper limit of how high amplitude each wavelet can become
    # X - the CONTINUOUS 1D synthetic time series data <--- THIS IS THE NON-NOISY DATA TO BE COMPARED TO




def real_shift_invariant_kmeans(windows, K, P, method_idx, max_iter=10, rng=None):


# CLUSTERING STEP NOT USING GROUND TRUTH ------------------------------------------------------------------------------
    if rng is None:
        rng = np.random.default_rng()
       
    n_samples, L = windows.shape
    centroid_length = P


    # Randomly picks indices using rng
    random_indices = rng.choice(n_samples, size=K, replace=False)


    # Sets shape of centroids to (K,P) = (Num of clusters, length of each centroid)
    centroids = np.zeros((K, centroid_length))


    # Initalizes each centroid by selecting a random P slice from the windows data
    for i, idx in enumerate(random_indices):
        # Selects a random starting index with a range [0, L- P + 1] to make sure P stays in the bounds of the window
        start_idx = rng.integers(0, L - P + 1)
        # Assigns inital centroids
        centroids[i] = windows[idx, start_idx:start_idx+P]
   
    # Iterative loop start - Runs k-means clustering for a maximum if max_iter iteration or until convergence
    for iteration in range(max_iter):
        # Finds current labels, shifts, and distances using the CURRENT estimated centroids and NOT C_true (Ground Truth)
        labels, shifts, distances = wrappers.si_pairwise_distances_argmin_min(
            windows, centroids, 'cosine', []
        )
       
        # Counts the number of points currently in each cluster (This will change with each iteration)
        cluster_ids, cluster_size = np.unique(labels, return_counts=True)
       
        # Create a buffer for the newly calculated centroids (This will be filled with values in the update step)
        new_centroids = np.zeros_like(centroids)
       
        # END OF NEW CLUSTERING CODE -------------------------------------------------------------------------------


        # This is where the six functions differ

        if method_idx == 0:  # Default averaging
            for sample_id, sample in enumerate(windows):
                cluster_id = labels[sample_id]
                shift = shifts[sample_id]
                new_centroids[cluster_id] += sample[shift:shift+centroid_length]
               
        elif method_idx == 1:  # Coefficient weighted average (using NNLS)
            for sample_id, sample in enumerate(windows):
                cluster_id = labels[sample_id]
                shift = shifts[sample_id]
                x_shifted = sample[shift:shift+centroid_length]
                # NNLS Step
                coef, _ = nnls(centroids[cluster_id][:, np.newaxis], x_shifted)
                new_centroids[cluster_id] += x_shifted * coef
               
        elif method_idx == 2:  # rank-1 SVD
            X_shifted = np.zeros((n_samples, centroid_length))
            for sample_id, sample in enumerate(windows):
                shift = shifts[sample_id]
                X_shifted[sample_id] = sample[shift:shift+centroid_length]
            for cluster_id in cluster_ids:
                members = (labels == cluster_id)
                if np.sum(members) == 1:
                    new_centroids[cluster_id] = X_shifted[members]
                else:
                    #SVD Step
                    coef, _, new_centroids[cluster_id] = svds(X_shifted[members], k=1)
                    new_centroids[cluster_id] *= np.sign(np.mean(coef))
                   
        elif method_idx == 3:  #rank-1 SVD with spherical normalization
            X_shifted = np.zeros((n_samples, centroid_length))
            for sample_id, sample in enumerate(windows):
                shift = shifts[sample_id]
                x_shifted = sample[shift:shift+centroid_length]
                # Normalization step
                X_shifted[sample_id] = x_shifted / (np.sqrt(np.sum(x_shifted**2)) + 1e-12)
            for cluster_id in cluster_ids:
                members = (labels == cluster_id)
                if np.sum(members) == 1:
                    new_centroids[cluster_id] = X_shifted[members]
                else:
                    #SVD Step
                    coef, _, new_centroids[cluster_id] = svds(X_shifted[members], k=1)
                    new_centroids[cluster_id] *= np.sign(np.mean(coef))
                   
        elif method_idx == 4:  # L2 Normalized versions of NNLS
            for sample_id, sample in enumerate(windows):
                cluster_id = labels[sample_id]
                shift = shifts[sample_id]
                x_shifted = sample[shift:shift+centroid_length]
                # Normalization step
                x_norm = np.sqrt(np.sum(x_shifted**2)) + 1e-12
                x_shifted_normalized = x_shifted / x_norm
                # NNLS step
                coef, _ = nnls(centroids[cluster_id][:, np.newaxis], x_shifted_normalized)
                new_centroids[cluster_id] += x_shifted_normalized * coef
           
        else:  # Default averaging with L2 normalization
            X_shifted = np.zeros((n_samples, centroid_length))
            for sample_id, sample in enumerate(windows):
                shift = shifts[sample_id]
                x_shifted = sample[shift:shift+centroid_length]
                X_shifted[sample_id] = x_shifted / (np.sqrt(np.sum(x_shifted**2)) + 1e-12)
            for cluster_id in cluster_ids:
                members = (labels == cluster_id)
                if np.sum(members) == 1:
                    new_centroids[cluster_id] = X_shifted[members]
                else:
                    new_centroids[cluster_id] = np.mean(X_shifted[members], axis=0)




        # Re-normalizes centroids before assigning in the next loop iteration
        norm = np.sqrt(np.sum(new_centroids**2, axis=1, keepdims=True))
        centroids = np.divide(new_centroids, norm, out=np.zeros_like(new_centroids), where=norm!=0)
       
    return centroids, labels, shifts


# SHIFT INVARIENT function to find the cosine_similarity between ground truth and new centroids.
def cosine_sim(C_true, centroids):
    cosmax = np.zeros((len(C_true), len(centroids)))
    njs = np.sqrt(np.sum(centroids**2, axis=1))
    for i in range(len(C_true)):
        ni = np.sqrt(np.sum(C_true[i]**2))
        ctrue_i = np.flip(C_true[i])
        for j in range(len(centroids)):
            cosmax[i, j] = np.max(np.convolve(ctrue_i, centroids[j])) / (ni * njs[j])
    return cosmax


# MAIN CODE


method_names = ['ave', 'nnls', 'svd', 'sph-svd', 'sph-nnls', 'sph-ave']
n_methods = len(method_names)
n_noise = 5
n_perfs = 5
n_runs = 10


perf_stats = np.zeros((n_methods, n_perfs, n_noise, n_runs))
all_alignments = -np.ones((n_methods, K_true, n_noise, n_runs))
all_rates = np.zeros((K_true, n_runs))
all_highs = np.zeros((K_true, n_runs))


for run_index in range(n_runs):
    rng = np.random.default_rng(run_index)


    # C_true is generated in this function call
    C_true, rates, highs, x = generate_time_series_with_dict(P, K_true, rate_lim, height_min, height_max_lim, rng)
    all_rates[:, run_index] = rates
    all_highs[:, run_index] = highs


    # C_norm is created by normalizing C_true
    C_norm = C_true / np.sqrt(np.sum(C_true**2, axis=1, keepdims=True))

    


    # Noise sweep function
    for noise_factor in range(n_noise):
        noise_level = noise_factor / 20 * np.std(x)
        windows_clean = x[:int(len(x)/L)*L].reshape([-1, L])
        windows = windows_clean + noise_level * rng.normal(size=windows_clean.shape)


        # Noise level print
        print(f"\n--- Run {run_index} | Noise Level {noise_level:.4f} ---")


        for method_idx, method_name in enumerate(method_names):
            # K-Means clustering function call
            centroids, labels, shifts = real_shift_invariant_kmeans(
                windows, K=K_true, P=P, method_idx=method_idx, max_iter=10, rng=rng
            )


            # # Take the maximum cosine similarity for each true wavelet
            cosmax = cosine_sim(C_norm, centroids)
            alignments = np.max(cosmax, axis=1)  


            sse = np.zeros(windows.shape[0])
            nnsse = np.zeros(windows.shape[0])
           
            for sample_id, sample in enumerate(windows):
                cluster_id = labels[sample_id]
                shift = shifts[sample_id]
                x_shifted = sample[shift:shift + P]
               
                # Standard SSE
                coef = np.sum(centroids[cluster_id] * x_shifted)
                sse[sample_id] = (np.sum(sample[:shift]**2) +
                                  np.sum(sample[shift+P:]**2) +
                                  np.sum((x_shifted - coef * centroids[cluster_id])**2))
               
                # NNLS SSE
                coef_nnls, _ = nnls(centroids[cluster_id][:, np.newaxis], x_shifted)
                nnsse[sample_id] = (np.sum(sample[:shift]**2) +
                                    np.sum(sample[shift+P:]**2) +
                                    np.sum((x_shifted - coef_nnls * centroids[cluster_id])**2))


            rmse_sse = np.sqrt(np.mean(sse))
            rmse_nnsse = np.sqrt(np.mean(nnsse))
           
           # Prints min, mean, max, SSE, and NNSSE for all six methods
            print(f"{method_name:<12} | Min: {np.min(alignments):.3f} | Mean: {np.mean(alignments):.3f} | Max: {np.max(alignments):.3f} | SSE: {rmse_sse:.3f} | NNSSE: {rmse_nnsse:.3f}")
           
            all_alignments[method_idx, :, noise_factor, run_index] = alignments
            perf_stats[method_idx, :, noise_factor, run_index] = np.array([
                np.min(alignments), np.mean(alignments), np.max(alignments), rmse_sse, rmse_nnsse
            ])
