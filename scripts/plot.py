import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

thput_base = {}
def extract_thput_base(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    for line in data:
        # Regex to match your data format
        match = re.match(r"([A-Za-z_+\s\S]+): Element ([0-9]+) bytes \| ([0-9]+\.[0-9]+) MBPS \| [0-9]+ ns", line.strip())
        if match:
            dev = match.group(1)
            size = match.group(2)
            thput = float(match.group(3))
            if dev not in thput_base:
                thput_base[dev] = {}
            thput_base[dev][size] = thput

LIMIT = 25.4 * 1000
def main():
    file = sys.argv[1]
    extract_thput_base(file)

    plt.figure(figsize=(8, 4))
    dev = next(iter(thput_base))
    thput_base["LIMIT"] = {}
    for size in thput_base[dev].keys():
        thput_base["LIMIT"][size] = LIMIT
    for dev in thput_base.keys():
        plt.plot(thput_base[dev].keys(), thput_base[dev].values(), '.-', label=dev)

    print("Size", end="\t")
    for dev in thput_base.keys():
        print(dev, end="\t")
    print()
    for size in thput_base[dev].keys():
        print(size, end="\t")
        for dev in thput_base.keys():
            print(thput_base[dev][size], end="\t")
        print()
    
    plt.xlabel('Element size (Bytes)')  # Label the x-axis
    plt.ylabel('Throughput (MBPS)')  # Label the x-axis
    #plt.yscale('log')
    plt.legend()

    plt.savefig(sys.argv[2] + ".png")  # Save the chart to a file
    

if __name__ == "__main__":
    main()
