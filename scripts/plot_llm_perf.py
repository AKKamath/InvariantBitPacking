import sys
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
from matplotlib.patches import Patch


def parse_input_file(input_file):
    records = []
    current_model = None

    with open(input_file, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("make[") or line.startswith("~/"):
                continue

            parts = [part.strip() for part in re.split(r"\t+|\s{2,}", line) if part.strip() != ""]
            if len(parts) < 3:
                continue

            if parts[1] == "Cache time (s)" and parts[2] == "Inference time (s)":
                current_model = parts[0]
                continue

            if current_model is None:
                continue

            method = parts[0]
            cache_time = float(parts[1])
            inference_time = float(parts[2])
            records.append(
                {
                    "model": current_model,
                    "method": method,
                    "cache": cache_time,
                    "inference": inference_time,
                }
            )

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

    method_order = ["InfiniGen + IBP", "InfiniGen"]
    methods_found = []
    for preferred in method_order:
        if any(preferred in methods_by_model[model] for model in models):
            methods_found.append(preferred)
    for model in models:
        for method in methods_by_model[model]:
            if method not in methods_found:
                methods_found.append(method)

    fig, ax = plt.subplots(figsize=(9, 5))

    bar_width = 0.34
    y = np.arange(len(models))
    offsets = np.linspace(-bar_width / 2, bar_width / 2, max(2, len(methods_found)))

    method_colors = {
        "InfiniGen": {"cache": "#e67e22", "inference": "#f8c291"},
        "InfiniGen + IBP": {"cache": "#1e8449", "inference": "#a9dfbf"},
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
    ax.set_ylabel("Model", fontweight="bold", fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(
            facecolor=method_colors.get("InfiniGen", fallback_colors)["cache"],
            edgecolor="black",
            label="Infinigen (Cache)",
        ),
        Patch(
            facecolor=method_colors.get("InfiniGen + IBP", fallback_colors)["cache"],
            edgecolor="black",
            label="Infinigen + IBP (Cache)",
        ),
        Patch(
            facecolor=method_colors.get("InfiniGen", fallback_colors)["inference"],
            edgecolor="black",
            label="Infinigen (Inference)",
        ),
        Patch(
            facecolor=method_colors.get("InfiniGen + IBP", fallback_colors)["inference"],
            edgecolor="black",
            label="Infinigen + IBP (Inference)",
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
