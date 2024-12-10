import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

comp = {}
ctr = 0
comp[ctr] = {}
total_nodes = 0
cur_centroids = 0
def extract_thput_base(file_path):
    global ctr
    global cur_centroids
    with open(file_path, 'r') as file:
        data = file.readlines()
    for line in data:
        # Regex to match your data format
        match = re.match(r"Num nodes: ([0-9]+), num_centroids ([0-9]+), used centroids ([0-9]+)", line.strip())
        if match:
            total_nodes = float(match.group(1))
        match = re.match(r"Centroids ([0-9]+), compressed: ([0-9.]+)", line.strip())
        if match:
            centroids = float(match.group(1))
            # Compressed - Number of centroids * 2 [1 mask, 1 bitval] / total_nodes * 100 [for percent]
            comp_perc = float(match.group(2)) * 100 - centroids * 2 / float(total_nodes) * 100
            print(f"{int(centroids)}\t{comp_perc:.3f}\t{float(match.group(2)) * 100:.3f}")
            if centroids < cur_centroids:
                ctr += 1
                comp[ctr] = {}
            cur_centroids = centroids
            comp[ctr][centroids] = comp_perc

labels = ["Asteroids.f32", "Reddit", "Uniform", "Normal"]

def main():
    file = sys.argv[1]
    extract_thput_base(file)

    plt.figure(figsize=(8, 2.5))
    for i in comp.keys():
        plt.plot(comp[i].keys(), comp[i].values(), '.-', label=labels[i])
    plt.xlabel('Clusters')  # Label the x-axis
    plt.ylabel('Net space saved (%)')  # Label the x-axis
    plt.xscale('log')
    #plt.yscale('symlog')
    plt.legend()

    plt.savefig(sys.argv[2] + ".png", bbox_inches='tight')  # Save the chart to a file
    

if __name__ == "__main__":
    main()  
