import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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




# Code to convert sound percents to SNR
noise_step = 0.20


def snr_label(noise_idx, noise_step=noise_step):
    # sigma_noise / sigma_signal
    ratio = noise_idx * noise_step
    if ratio == 0:
        return "No Noise"
    snr_db = -20 * np.log10(ratio)
    return f"{snr_db:.1f} dB"


noise_labels = [snr_label(i) for i in range(n_noise)]




# List to hold the row data
rows = []
# Loop through each combination in order of: Run #, Noise level #, and then method used
for run_idx in range(n_runs):
    for noise_idx in range(n_noise):
        for method_idx in range(n_methods):
            rows.append({
                'Run #': run_idx,
                'Noise Level #': noise_labels[noise_idx],                        
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


summary_table = (
    df.groupby(['Noise Level #', 'Method'])['Mean']
    .mean()
    .unstack(level='Method')
)


# Reorder columns to match inital table
summary_table = summary_table[method_names]


# Create Grand Average row with the averages all of the other noise level means
summary_table.loc['Grand Average'] = summary_table.mean(axis=0)


# Save summary table to CSV (includes Grand Average)
summary_table.to_csv('Clustering_Means.csv')






# Graph Creation


from scipy import stats as scipy_stats


# Build Grand Average version (per-run averages across noise levels)
grand_avg_df = (
    df.groupby(['Run #', 'Method'], observed=True)['Mean']
    .mean()
    .reset_index()
)
grand_avg_df['Noise Level #'] = 'Grand Average'


plot_df = pd.concat([df, grand_avg_df], ignore_index=True)


noise_order = noise_labels + ['Grand Average']
plot_df['Noise Level #'] = pd.Categorical(plot_df['Noise Level #'], categories=noise_order, ordered=True)


# Compute mean, and 95% CI half-width (t-distribution, n=10 runs) per Noise Level x Method
summary = (
    plot_df.groupby(['Noise Level #', 'Method'], observed=True)['Mean']
    .agg(['mean', 'sem', 'count'])
    .reset_index()
)
t_crit = scipy_stats.t.ppf(0.975, summary['count'] - 1)  # 95% two-sided
summary['ci95'] = t_crit * summary['sem']


plt.figure(figsize=(11, 6))
sns.set_theme(style="whitegrid")


x_positions = np.arange(len(noise_order))
n_methods_plot = len(method_names)


dodge_width = 0.6
offsets = np.linspace(-dodge_width / 2, dodge_width / 2, n_methods_plot)


colors = sns.color_palette(n_colors=n_methods_plot)


for i, method in enumerate(method_names):
    sub = summary[summary['Method'] == method].set_index('Noise Level #').reindex(noise_order)


    line, caps, bars = plt.errorbar(
        x_positions + offsets[i],
        sub['mean'].values,
        yerr=sub['ci95'].values,
        marker='o',
        linestyle='none',
        linewidth=None,
        capsize=3,
        label=method,
        color=colors[i],
        # no alpha here — line & markers stay fully opaque
    )


    # Fade only the error bars (vertical lines) and caps (horizontal ticks)
    for bar in bars:
        bar.set_alpha(0.75)
    for cap in caps:
        cap.set_alpha(0.75)


plt.xticks(x_positions, noise_order, rotation=45, ha='right', rotation_mode='anchor')
plt.title('Mean Cosine Similarity Across different SNRs (95% CI)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Signal to Noise Ratio (SNR) ', fontsize=12, labelpad=10)
plt.ylabel('Mean Cosine Similarity', fontsize=12, labelpad=10)
plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


plt.savefig('graph_option1_fixed.png', dpi=300)
plt.show()
