import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from matplotlib.lines import Line2D

# Imports the .npz file
data = np.load('2Clustering_Results.npz')

# Lists all arrays in .npz file
print(data.files)

# Main points of data to be used
# Size of (6, 196, 5, 10): (methods, wavelets, noise levels, runs)
stats = data['all_alignments']

method_names = ['ave', 'nnls', 'svd', 'sph-svd', 'sph-nnls', 'sph-ave']

# Gets mean and std for all 196 wavelets
# Has the shape: (6 methods, 5 noise levels, 10 runs)
mean_array = np.mean(stats, axis=1)
std_array = np.std(stats, axis=1)

n_methods, n_noise, n_runs = mean_array.shape
n_wavelets = stats.shape[1]

# Code to convert sound percents to SNR
noise_step = 0.20


def snr_label(noise_idx, noise_step=noise_step):
    ratio = noise_idx * noise_step
    if ratio == 0:
        return "No Noise"
    snr_db = -20 * np.log10(ratio)
    return f"{snr_db:.1f} dB"

noise_labels = [snr_label(i) for i in range(n_noise)]


# Function that helps generate error bars for minimum
# (Also tends to slow down the graph generation code)
def get_bootstrap_min_ci(df, n_boot=1000):
    ci_values = []
    for _, group in df.groupby(['Noise Level #', 'Method'], observed=True):
        vals = group['Alignment'].values
        if len(vals) == 0:
            ci_values.append(0)
            continue
        boot_mins = np.min(np.random.choice(vals, size=(n_boot, len(vals)), replace=True), axis=1)
        ci_lower = np.percentile(boot_mins, 2.5)
        ci_upper = np.percentile(boot_mins, 97.5)
        ci_values.append((ci_upper - ci_lower) / 2)
    return ci_values


# List to hold the row data for run-level statistics
rows = []
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

# Exports Clustering_Results_Spreadsheet.csv
df = pd.DataFrame(rows)
df.to_csv('2Clustering_Results_Spreadsheet.csv', index=False)

print("\nCSV file created")
print(df.head())

# Exports Clustering_Means.csv
summary_table = (
    df.groupby(['Noise Level #', 'Method'])['Mean']
    .mean()
    .unstack(level='Method')
)
summary_table = summary_table[method_names]
summary_table.loc['Grand Average'] = summary_table.mean(axis=0)
summary_table.to_csv('2Clustering_Means.csv')


# Data gathering
stats_transposed = np.transpose(stats, (0, 2, 3, 1))

idx = pd.MultiIndex.from_product(
    [method_names, noise_labels, range(n_runs), range(n_wavelets)],
    names=['Method', 'Noise Level #', 'Run #', 'Wavelet #']
)

df_raw = pd.DataFrame({'Alignment': stats_transposed.flatten()}, index=idx).reset_index()

# Build Grand Average dataset using all raw input data points
grand_avg_df = df_raw.copy()
grand_avg_df['Noise Level #'] = 'Grand Average'

plot_df = pd.concat([df_raw, grand_avg_df], ignore_index=True)

noise_order = noise_labels + ['Grand Average']
plot_df['Noise Level #'] = pd.Categorical(plot_df['Noise Level #'], categories=noise_order, ordered=True)

# Compute mean, sem, count, 25th/10th percentiles, and minimum from all data points
summary = (
    plot_df.groupby(['Noise Level #', 'Method'], observed=True)['Alignment']
    .agg(
        mean='mean',
        sem='sem',
        count='count',
        std='std',
        # 25th percentile
        q25=lambda x: np.percentile(x, 25),
        # 10th percentile 
        q10=lambda x: np.percentile(x, 10), 
        # Min value
        min_val='min'                       
    )
    .reset_index()
)

# 95% Confidence Interval across raw points for the mean
t_crit = scipy_stats.t.ppf(0.975, summary['count'] - 1)
summary['ci95_mean'] = t_crit * summary['sem']

# 95% confident interval for 25% and 10% quartile points (Values come from the density of the normal distribution at the 25th and 10th percentiles)
summary['ci95_q25'] = t_crit * (1.36 * summary['std'] / np.sqrt(summary['count']))
summary['ci95_q10'] = t_crit * (1.71 * summary['std'] / np.sqrt(summary['count']))

# Compute realistic 95% CI for minimums
summary['ci95_min'] = get_bootstrap_min_ci(plot_df)


# Plotting

plt.figure(figsize=(11, 6))
sns.set_theme(style="whitegrid")

x_positions = np.arange(len(noise_order))
n_methods_plot = len(method_names)

dodge_width = 0.6
offsets = np.linspace(-dodge_width / 2, dodge_width / 2, n_methods_plot)
colors = sns.color_palette(n_colors=n_methods_plot)

for i, method in enumerate(method_names):
    sub = summary[summary['Method'] == method].set_index('Noise Level #').reindex(noise_order)

    # Mean + 95% confidence interval - ('o' marker)
    line, caps, bars = plt.errorbar(
        x_positions + offsets[i],
        sub['mean'].values,
        yerr=sub['ci95_mean'].values,
        marker='o',
        linestyle='none',
        capsize=3,
        label=method,
        color=colors[i],
    )

    for bar in bars:
        bar.set_alpha(0.75)
    for cap in caps:
        cap.set_alpha(0.75)

    # 25th Percentile marker - ('v' marker)
    plt.errorbar(
        x_positions + offsets[i],
        sub['q25'].values,
        yerr=sub['ci95_q25'].values,
        marker='v',
        linestyle='none',
        capsize=3,
        color=colors[i],
        alpha=0.8,
        markersize=6
    )

    # 10th Percentile marker - ('s' marker)
    plt.errorbar(
        x_positions + offsets[i],
        sub['q10'].values,
        yerr=sub['ci95_q10'].values,
        marker='s',
        linestyle='none',
        capsize=3,
        color=colors[i],
        alpha=0.7,
        markersize=6
    )

    # Minimum value marker - ('^' marker)
    plt.errorbar(
        x_positions + offsets[i],
        sub['min_val'].values,
        yerr=sub['ci95_min'].values,
        marker='^',
        linestyle='none',
        capsize=3,
        color=colors[i],
        alpha=0.5,
        markersize=6
    )


# Graph legend for each marker

method_legend = plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

shape_legend_elements = [
    Line2D([0], [0], marker='o', color='gray', label='Mean', linestyle='None', markersize=7),
    Line2D([0], [0], marker='v', color='gray', label='25th Percentile', linestyle='None', markersize=7),
    Line2D([0], [0], marker='s', color='gray', label='10th Percentile', linestyle='None', markersize=7),
    Line2D([0], [0], marker='^', color='gray', label='Minimum', linestyle='None', markersize=7),
]

shape_legend = plt.legend(
    handles=shape_legend_elements,
    title='Marker Shape',
    bbox_to_anchor=(1.05, 0.55),
    loc='upper left'
)

# Re-add the Method legend so matplotlib doesn't overwrite it
plt.gca().add_artist(method_legend)
plt.xticks(x_positions, noise_order, rotation=45, ha='right', rotation_mode='anchor')
plt.title('Mean Cosine Similarity Across different SNRs (95% CI)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Signal to Noise Ratio (SNR)', fontsize=12, labelpad=10)
plt.ylabel('Mean Cosine Similarity', fontsize=12, labelpad=10)
plt.tight_layout()
plt.savefig('3graph_option1_fixed.png', dpi=300)
plt.show()
