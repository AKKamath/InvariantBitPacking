import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

comp = {}
ctr = 0
comp[ctr] = {}
total_nodes = 0
def extract_thput_base(file_path):
    global ctr
    with open(file_path, 'r') as file:
        data = file.readlines()
    for line in data:
        # Regex to match your data format
        match = re.match(r"Num nodes: ([0-9]+), num_centroids ([0-9]+), used centroids ([0-9]+)", line.strip())
        if match:
            total_nodes = float(match.group(1))
        match = re.match(r"Centroids ([0-9]+), compressed: ([0-9.]+)", line.strip())
        if match:
            # Number of centroids * 2 [1 mask, 1 bitval] / total_nodes * 100 [for percent]
            centroids = float(match.group(1)) * 2 / float(total_nodes) * 100
            # Compressed - overhead of centroids
            comp_perc = float(match.group(2)) * 100 - centroids
            print(f"{centroids:.3f}\t{float(match.group(2)) * 100:.3f}")
            if(centroids in comp[ctr]):
                ctr += 1
                comp[ctr] = {}
            comp[ctr][centroids] = comp_perc

labels = ["Uniform", "Normal", "Zipf"]

def main():
    file = sys.argv[1]
    extract_thput_base(file)

    plt.figure(figsize=(8, 2.5))
    for i in comp.keys():
        plt.plot(comp[i].keys(), comp[i].values(), '.-')
    plt.xlabel('Cluster overhead (%)')  # Label the x-axis
    plt.ylabel('Net space saved (%)')  # Label the x-axis
    #plt.xscale('symlog')
    #plt.yscale('symlog')
    plt.legend()

    plt.savefig(sys.argv[2] + ".pdf", bbox_inches='tight')  # Save the chart to a file
    

if __name__ == "__main__":
    main()  
