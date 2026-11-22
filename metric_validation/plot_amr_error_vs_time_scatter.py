import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def get_data_from_file(filename):
    """Parses a YAML log file and returns a DataFrame."""
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)

    plot_data = []
    for key, value in data.items():
        if 'total_time' in value and 'final_rel_l1' in value:
            plot_data.append({
                'total_time': value['total_time'],
                'final_rel_l1': value['final_rel_l1']
            })
    return pd.DataFrame(plot_data)


# --- Configuration ---
parser = argparse.ArgumentParser(
    description="Scatter total_time against final_rel_l1 for the seven Table 3 logs."
)
parser.add_argument(
    "logs",
    nargs=7,
    metavar="LOG",
    help=(
        "seven log paths in Table 3 order: "
        "Ours AMR, Ours without AMR, Obs AMR, Obs without AMR, "
        "Uni 4k AMR, Uni 4k without AMR, Uni 12k without AMR"
    ),
)
args = parser.parse_args()
labels = [
    "Ours w AMR",
    "Ours w/o AMR",
    "Obs w AMR",
    "Obs w/o AMR",
    "Uni w AMR",
    "Uni (4K) w/o AMR",
    "Uni (12K) w/o AMR",
]
files_to_plot = list(zip(args.logs, labels))

# Define lists for colors and markers
colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
          '#d62728', '#9467bd', '#8c564b', '#e377c2']
markers = ['o', 's', '^', 'D', 'v', 'h', 'p', '8', '*']

# --- Plotting ---
plt.figure(figsize=(10, 6))

plt.yscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.6)

ax = plt.gca()
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=20))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

# Unpack the tuple (filepath, label_name) here
for i, (filepath, label_name) in enumerate(files_to_plot):
    # 1. Load Data
    try:
        df = get_data_from_file(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        continue

    # 2. Select Style
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]

    # 3. Plot Main Scatter Points (Individual Samples)
    plt.scatter(
        df['total_time'],
        df['final_rel_l1'],
        alpha=0.5,         # Lower alpha (transparency) for background points
        label=label_name,  # Label applies to these points for the legend
        color=color,
        marker=marker,
        s=10
    )

    # 4. Calculate and Highlight Average Point
    if not df.empty:
        avg_time = df['total_time'].mean()
        avg_l1 = df['final_rel_l1'].mean()

        plt.scatter(
            avg_time,
            avg_l1,
            color=color,
            marker=marker,
            s=400,              # Larger size for the average point
            edgecolors='black',  # Black outline to highlight it
            linewidth=1.5,
            alpha=1.0           # Fully opaque
        )

# --- Formatting ---
plt.title("Relative L1 Error & Time Scatter", fontsize=30)
plt.xlabel('Time (s)', fontsize=30)
plt.ylabel('Relative L1 Error (Log)', fontsize=30)

# Legend settings
plt.legend(loc='best', fontsize=20, markerscale=5, framealpha=1)

plt.tight_layout()

output_filename = 'amr_error_vs_time_scatter.jpg'
plt.savefig(output_filename, format='jpg', dpi=300,
            bbox_inches='tight', pad_inches=0.05)
plt.show()
