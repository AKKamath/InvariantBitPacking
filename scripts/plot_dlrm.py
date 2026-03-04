import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

ids = [
    '512B',
    '1024B',
    '2048B',
]
colors = [
    'tab:green',
    'tab:orange',
    'tab:purple',
    'tab:cyan',
    'tab:blue',
    'black',
]


def _parse_block_speedup(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    rows = []
    i = 0
    while i < len(lines):
        header = lines[i]
        if not header.lower().startswith('batch '):
            i += 1
            continue

        parts = re.split(r'\t+', header)
        if len(parts) < 4:
            i += 1
            continue

        batch_match = re.search(r'Batch\s+(\d+)', parts[0], flags=re.IGNORECASE)
        if batch_match is None:
            i += 1
            continue

        batch = batch_match.group(1)

        metric_names = []
        for token in parts[1:4]:
            token_clean = token.strip()
            metric_match = re.match(r'(.+?)\s+speedup$', token_clean, flags=re.IGNORECASE)
            metric_names.append(metric_match.group(1) if metric_match else token_clean)

        if i + 1 >= len(lines):
            break

        value_tokens = [token for token in re.split(r'\s+', lines[i + 1].strip()) if token]
        if len(value_tokens) < 3:
            i += 1
            continue

        try:
            values = [float(value_tokens[0]), float(value_tokens[1]), float(value_tokens[2])]
        except ValueError:
            i += 1
            continue

        row = {'Batch': batch}
        row.update({name: val for name, val in zip(metric_names, values)})
        rows.append(row)
        i += 2

    if not rows:
        return None

    df = pd.DataFrame(rows)
    expected_cols = ['Batch'] + ids
    if all(col in df.columns for col in expected_cols):
        return df[expected_cols]
    return df


def _parse_legacy_tsv(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df.loc[:, ~df.columns.astype(str).str.contains(r'^Unnamed')]
    df.columns = df.columns.astype(str).str.strip()

    rename_map = {'1KB': '1024B', '2KB': '2048B'}
    df = df.rename(columns=rename_map)

    if 'Batch' not in df.columns:
        df.insert(0, 'Batch', [str(x) for x in [1024, 2048, 4096, 8192][:len(df)]])
    else:
        df['Batch'] = df['Batch'].astype(str).str.extract(r'(\d+)')[0].fillna(df['Batch'].astype(str))

    keep_cols = ['Batch'] + [col for col in ids if col in df.columns]
    return df[keep_cols]


def load_plot_df(file_path):
    df = _parse_block_speedup(file_path)
    if df is None:
        df = _parse_legacy_tsv(file_path)

    for col in ids:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in input file: {file_path}")

    df = df[['Batch'] + ids].copy()
    for col in ids:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=ids)
    return df

def main():
    file = sys.argv[1]
    df = load_plot_df(file)

    #plt.figure(figsize=(8, 2.5))
    fig, ax = plt.subplots(figsize=(9, 2.5))

    width = 0.2
    x_pos = np.arange(0, len(df.index), 1)

    max_val = 0
    for i, cfg in enumerate(ids):
        bars = ax.bar(x_pos + i * width, df[cfg], width, color=colors[i], label=cfg, alpha=0.7, edgecolor='black')
        max_val = max(max_val, max(df[cfg]))
        #for bar in bars:
        #    yval = bar.get_height()
        #    ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom', fontsize=12)
    plt.xticks(x_pos + width, df['Batch'].astype(str).tolist(), fontsize=11.5)
    plt.yticks(np.arange(0, np.ceil(max_val) + 1, 1), fontsize=12)
    #plt.yticks(np.arange(0, 43, 6), fontsize=28)
    ax.grid(axis='y', linestyle='--')
    ax.yaxis.set_major_formatter('{x:.0f}x')
    plt.ylabel("Speedup", fontweight='bold', fontsize=13)
    plt.xlabel("Batch size", fontweight='bold', fontsize=13)
    plt.legend(loc='upper center', bbox_to_anchor=(0.47, 1.2), ncol=4, frameon=True, prop={'weight':'bold'}, fontsize=12, columnspacing=0.8)
    plt.savefig(sys.argv[2] + ".pdf", bbox_inches='tight', pad_inches=0)  # Save the chart to a file


if __name__ == "__main__":
    main()
