import numpy as np
import pandas as pd

# Imports the .npz file
data = np.load('Clustering_Results.npz')

# Lists all arrays in .npz file
print(data.files)

# Main points of data to be used
# Size of (6, 196, 5, 10)
# 6 - Which of the six methods used
# 196 - The Ground Truth wavelets being compared too
# 5 - The five different noise levels used during each run
# 10 - The ten runs for each randomized ground truth
stats = data['all_alignments']

method_names = ['ave', 'nnls', 'svd', 'sph-svd', 'sph-nnls', 'sph-ave']

# Gets mean and std for all 196 wavelets
# Has the shape: (6 methods, 5 noise levels, 10 runs)
mean_array = np.mean(stats, axis=1)
std_array = np.std(stats, axis=1)

n_methods, n_noise, n_runs = mean_array.shape

# List to hold the row data
rows = []
# Loop through each combination in order of: Run #, Noise level #, and then method used
for run_idx in range(n_runs):
    for noise_idx in range(n_noise):
        for method_idx in range(n_methods):
            rows.append({
                'Run #': run_idx,
                'Noise Level #': "Noise Level " + str(noise_idx + 1),                        
                'Method': method_names[method_idx], 
                'Mean': mean_array[method_idx, noise_idx, run_idx],
                'Std': std_array[method_idx, noise_idx, run_idx]
            })


# Converting to DataFrame and exporting as CSV
df = pd.DataFrame(rows)
df.to_csv('Clustering_Results_Spreadsheet.csv', index=False)

print("\nCSV file created")

# Prints the first few rows to preview
print(df.head())
