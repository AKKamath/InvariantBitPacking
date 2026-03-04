import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO

ids = [
    'dglcomp',
    'comp',
    'cpuonly',
    'cpuasync',
]
labels = {
    'dgl' : "DGL",
    'dglcomp' : "DGL+IBP(M)",
    'base' : "Legion",
    'comp' : "Legion+IBP(C)",
    'cpuonly' : "Legion+IBP(M)",
    'cpuasync' : "Legion+IBP(C/M)",
}
colors = [
    'tab:green',
    'tab:orange',
    'tab:purple',
    'tab:cyan',
    'tab:blue',
    'black',
]

dataset_display_names = {
    'pubmed_ls': 'PubmedSU',
    'citeseer_ls': 'CiteseerSU',
    'cora_ls': 'CoraSU',
    'reddit': 'Reddit',
    'products': 'Products',
    'mag': 'MAG',
    'geo': 'GEOMEAN',
}


def _clean_dataframe(df):
    df = df.loc[:, ~df.columns.astype(str).str.contains(r'^Unnamed')]
    df.columns = df.columns.astype(str).str.strip()
    return df


def _extract_tsv_block(file_path, header_name):
    with open(file_path, 'r') as file:
        lines = [line.rstrip('\n') for line in file]

    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith(f"{header_name}\t"):
            start_idx = i
            break

    if start_idx is None:
        return None

    block_lines = [lines[start_idx]]
    for line in lines[start_idx + 1:]:
        if not line.strip():
            break
        if '\t' not in line:
            break
        block_lines.append(line)

    if len(block_lines) <= 1:
        return None

    return '\n'.join(block_lines) + '\n'


def _read_tsv_block(file_path, header_name):
    block = _extract_tsv_block(file_path, header_name)
    if block is None:
        return None
    return _clean_dataframe(pd.read_csv(StringIO(block), sep='\t'))


def build_plot_df(file_path):
    speedup_raw_df = _read_tsv_block(file_path, 'Speedup')
    if speedup_raw_df is not None and 'Speedup' in speedup_raw_df.columns:
        raw_df = speedup_raw_df
        speedup_df = raw_df.set_index('Speedup')
        speedup_df.index = speedup_df.index.astype(str).str.strip()
        speedup_df = speedup_df.apply(pd.to_numeric, errors='coerce')
        speedup_df = speedup_df.dropna(axis=1, how='all').dropna(axis=0, how='all')

        plot_df = pd.DataFrame({cfg: speedup_df.loc[labels[cfg]] for cfg in ids})
        plot_df.loc['geo'] = np.exp(np.log(plot_df).mean())
        return plot_df

    # Old format compatibility: start from runtimes and normalize to speedup.
    runtime_raw_df = _read_tsv_block(file_path, 'Avg time (s)')
    if runtime_raw_df is None or 'Avg time (s)' not in runtime_raw_df.columns:
        raise ValueError("Could not find a valid 'Speedup' or 'Avg time (s)' table in the input file.")

    raw_df = runtime_raw_df
    raw_df = raw_df.set_index('Avg time (s)').T
    raw_df.index = raw_df.index.astype(str).str.strip()
    raw_df = raw_df.apply(pd.to_numeric, errors='coerce')

    plot_df = pd.DataFrame(index=raw_df.index)
    plot_df['dglcomp'] = raw_df['dgl'] / raw_df['dglcomp']
    plot_df['comp'] = raw_df['base'] / raw_df['comp']
    plot_df['cpuonly'] = raw_df['base'] / raw_df['cpuonly']
    plot_df['cpuasync'] = raw_df['base'] / raw_df['cpuasync']
    plot_df.loc['geo'] = np.exp(np.log(plot_df).mean())
    return plot_df


def format_dataset_name(dataset_name):
    key = str(dataset_name).strip().lower()
    return dataset_display_names.get(key, str(dataset_name))


def reorder_datasets(plot_df):
    preferred_order = [name for name in dataset_display_names if name in plot_df.index]
    remaining = [name for name in plot_df.index if name not in preferred_order]
    return plot_df.loc[preferred_order + remaining]

def main():
    file = sys.argv[1]
    df = build_plot_df(file)
    df = reorder_datasets(df)
    datasets = [format_dataset_name(i) for i in df.index.tolist()]

    #plt.figure(figsize=(8, 2.5))
    fig, ax = plt.subplots(figsize=(9, 2.5))

    width = 0.2
    x_pos = np.arange(0, len(df.index), 1)

    max_val = 0
    for i, cfg in enumerate(ids):
        ax.bar(x_pos + i * width, df[cfg], width, color=colors[i], label=labels[cfg], alpha=0.7, edgecolor='black')
        max_val = max(max_val, max(df[cfg]))
        #for bar in bars:
        #    yval = bar.get_height()
        #    ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom', fontsize=12)
    plt.xticks(x_pos + width, datasets, fontsize=11.5)
    plt.yticks(np.arange(0, np.ceil(max_val) + 1, 1), fontsize=12)
    #plt.yticks(np.arange(0, 43, 6), fontsize=28)
    ax.grid(axis='y', linestyle='--')
    ax.yaxis.set_major_formatter('{x:.0f}x')
    plt.ylabel("Speedup", fontweight='bold', fontsize=13)
    plt.xlabel("Dataset", fontweight='bold', fontsize=13)
    plt.legend(loc='upper center', bbox_to_anchor=(0.47, 1.2), ncol=4, frameon=True, prop={'weight':'bold'}, fontsize=12, columnspacing=0.8)
    plt.savefig(sys.argv[2] + ".pdf", bbox_inches='tight', pad_inches=0)  # Save the chart to a file


if __name__ == "__main__":
    main()
