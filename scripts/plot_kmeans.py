import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

comp = {}
ctr = 0
comp[ctr] = {}
def extract_thput_base(file_path):
    global ctr
    with open(file_path, 'r') as file:
        data = file.readlines()
    for line in data:
        # Regex to match your data format
        match = re.match(r"Centroids ([0-9]+), compressed: ([0-9.]+)", line.strip())
        if match:
            centroids = float(match.group(1)) / float(32 * 1024) * 100
            comp_perc = float(match.group(2)) * 100
            if(centroids in comp[ctr]):
                ctr += 1
                comp[ctr] = {}
            comp[ctr][centroids] = comp_perc

labels = ["Uniform", "Normal", "Zipf"]

def main():
    file = sys.argv[1]
    extract_thput_base(file)

    plt.figure(figsize=(8, 4))
    for i in comp.keys():
        plt.plot(comp[i].keys(), comp[i].values(), '.-', label=labels[i])
    plt.xlabel('Cluster overhead (%)')  # Label the x-axis
    plt.ylabel('Space saved (%)')  # Label the x-axis
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.legend()

    plt.savefig(sys.argv[2] + ".png", bbox_inches='tight')  # Save the chart to a file
    

if __name__ == "__main__":
    main()
