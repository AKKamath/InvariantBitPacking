import sys
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
from matplotlib.patches import Patch


def parse_input_file(input_file):
    records = []
    current_model = None

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.startswith("make[") or line.startswith("~/"):
            continue

        method_match = re.match(r'^(FlexGen[^:]*?):\s*$', line)
        if method_match:
            method = method_match.group(1).strip()
            total_time = None
            weight_load = None

            while i < len(lines):
                inner = lines[i].strip()
                i += 1
                if not inner:
                    break
                m = re.match(r'^Total:\s*([\d.]+)', inner)
                if m:
                    total_time = float(m.group(1))
                m = re.match(r'^Weight load time:\s*([\d.]+)', inner)
                if m:
                    weight_load = float(m.group(1))

            if total_time is not None and weight_load is not None:
                records.append({
                    "model": "",
                    "method": method,
                    "cache": weight_load,
                    "inference": total_time - weight_load,
                })
            continue

        if not re.match(r'^(Total:|Cache time:|Weight load time:|Prefill:|Decode:)', line):
            current_model = line

    if not records:
        raise ValueError("No valid records found in input file.")

    return records


def main():
    input_file = sys.argv[1]
    output_prefix = sys.argv[2]

    rows = parse_input_file(input_file)

    models = []
    methods_by_model = defaultdict(dict)
    for row in rows:
        model = row["model"]
        if model not in methods_by_model:
            models.append(model)
        methods_by_model[model][row["method"]] = row

    method_order = ["FlexGen + IBP", "FlexGen"]
    methods_found = []
    for preferred in method_order:
        if any(preferred in methods_by_model[model] for model in models):
            methods_found.append(preferred)
    for model in models:
        for method in methods_by_model[model]:
            if method not in methods_found:
                methods_found.append(method)

    fig, ax = plt.subplots(figsize=(9, 3))

    bar_width = 0.34
    y = np.arange(len(models))
    offsets = np.linspace(-bar_width / 2, bar_width / 2, max(2, len(methods_found)))

    method_colors = {
        "FlexGen": {"cache": "#e67e22", "inference": "#f8c291"},
        "FlexGen + IBP": {"cache": "#1e8449", "inference": "#a9dfbf"},
    }
    fallback_colors = {
        "cache": "#7f7f7f",
        "inference": "#b0b0b0",
    }

    for method_index, method in enumerate(methods_found):
        cache_vals = []
        infer_vals = []
        for model in models:
            row = methods_by_model[model].get(method)
            if row is None:
                cache_vals.append(0.0)
                infer_vals.append(0.0)
            else:
                cache_vals.append(row["cache"])
                infer_vals.append(row["inference"])

        cache_vals = np.array(cache_vals)
        infer_vals = np.array(infer_vals)
        ypos = y + offsets[method_index]

        ax.barh(
            ypos,
            cache_vals,
            bar_width,
            color=method_colors.get(method, fallback_colors)["cache"],
            edgecolor="black",
            alpha=0.9,
        )
        ax.barh(
            ypos,
            infer_vals,
            bar_width,
            left=cache_vals,
            color=method_colors.get(method, fallback_colors)["inference"],
            edgecolor="black",
            alpha=0.9,
        )

    ax.set_xlim(0, max(1.0, ax.get_xlim()[1]))
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Time (s)", fontweight="bold", fontsize=12)
    #ax.set_ylabel("Model", fontweight="bold", fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(
            facecolor=method_colors.get("FlexGen", fallback_colors)["cache"],
            edgecolor="black",
            label="FlexGen (Weight load)",
        ),
        Patch(
            facecolor=method_colors.get("FlexGen + IBP", fallback_colors)["cache"],
            edgecolor="black",
            label="FlexGen + IBP (Weight load)",
        ),
        Patch(
            facecolor=method_colors.get("FlexGen", fallback_colors)["inference"],
            edgecolor="black",
            label="FlexGen (Inference)",
        ),
        Patch(
            facecolor=method_colors.get("FlexGen + IBP", fallback_colors)["inference"],
            edgecolor="black",
            label="FlexGen + IBP (Inference)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.24),
        ncol=2,
        frameon=True,
        fontsize=10,
        prop={'weight':'bold'},
    )
    plt.tight_layout()
    plt.savefig(output_prefix + ".pdf", bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
