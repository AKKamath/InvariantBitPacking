import re
import os
import sys
import numpy as np
import math
import glob

RATIOS = {}
THPUTS = {}
def extract_ratio(file_path, folder_path):
    dataset = re.findall(folder_path + '/(.*).log', file_path)[0]
    layer = -1
    with open(file_path, 'r') as file:
        for line in file.readlines():
            matches = re.findall(r'(.*): Uncompressed bytes: [0-9]+, compressed bytes: [0-9]+, ratio: ([0-9]+\.[0-9]+)', line)
            layer_num = re.findall(r'Layer ([0-9]+)', line)
            if len(layer_num) > 0:
                layer = int(layer_num[0])
            for match in matches:
                if match[0] not in RATIOS:
                    RATIOS[match[0]] = {}
                ratio = float(match[1])
                RATIOS[match[0]][dataset] = RATIOS[match[0]].get(dataset, []) + [ratio]
                if layer != -1:
                    RATIOS[match[0]][dataset + "_min"] = min(RATIOS[match[0]].get(dataset + "_min", math.inf), ratio)
                    RATIOS[match[0]][dataset + "_max"] = max(RATIOS[match[0]].get(dataset + "_max", -1), ratio)

def extract_thput(file_path, folder_path):
    dataset = re.findall(folder_path + '/(.*).log', file_path)[0]
    layer = -1
    with open(file_path, 'r') as file:
        for line in file.readlines():
            matches = re.findall(r'(.*): Time taken to decompress: ([0-9]+\.[0-9]+) ms. Throughput: ([0-9]+\.[0-9]+) MB/s', line)
            layer_num = re.findall(r'Layer ([0-9]+)', line)
            if len(layer_num) > 0:
                layer = int(layer_num[0])
            for match in matches:
                if match[0] not in THPUTS:
                    THPUTS[match[0]] = {}
                thput = float(match[2])
                THPUTS[match[0]][dataset] = THPUTS[match[0]].get(dataset, []) + [thput]
                if layer != -1:
                    THPUTS[match[0]][dataset + "_min"] = min(THPUTS[match[0]].get(dataset + "_min", math.inf), thput)
                    THPUTS[match[0]][dataset + "_max"] = max(THPUTS[match[0]].get(dataset + "_max", -1), thput)

# Example usage
folder_path = sys.argv[1]
datasets = sys.argv[2].split()

for dataset in datasets:
    extract_ratio(folder_path + '/' + dataset + '.log', folder_path)
    extract_thput(folder_path + '/' + dataset + '.log', folder_path)

print("Avg speedup", end='\t')
first_key = list(THPUTS.keys())[0]
for dataset in datasets:
    print("{:s}".format(dataset), end='\t')
print()

for algo in THPUTS:
    print("{:s}".format(algo), end='\t')
    for dataset in datasets:
        if(dataset in THPUTS[algo] and dataset in THPUTS["Transfer"]):
            if isinstance(THPUTS["Transfer"][dataset], list):
                transfer = sum(THPUTS["Transfer"][dataset]) / len(THPUTS["Transfer"][dataset])
            else:
                transfer = THPUTS["Transfer"][dataset]
            if isinstance(THPUTS[algo][dataset], list):
                thput = sum(THPUTS[algo][dataset]) / len(THPUTS[algo][dataset])
            else:
                thput = THPUTS[algo][dataset]
            print("{:.2f}".format(thput / transfer), end='\t')
        else:
            print("NA", end='\t')
    print()
print()

print("Space savings (%)", end='\t')
first_key = list(RATIOS.keys())[0]
datasets = RATIOS[first_key].keys()
for dataset in datasets:
    print("{:s}".format(dataset), end='\t')
print()

for algo in RATIOS:
    print("{:s}".format(algo), end='\t')
    for dataset in datasets:
        if(dataset in RATIOS[algo]):
            if isinstance(RATIOS[algo][dataset], list):
                print("{:.2f}".format((1.0 - (1.0 / (sum(RATIOS[algo][dataset]) / len(RATIOS[algo][dataset])))) * 100.0), end='\t')
            else:
                print("{:.2f}".format((1.0 - (1.0 / RATIOS[algo][dataset])) * 100.0), end='\t')

        else:
            print("NA", end='\t')
    print()
print()

'''
print("Thput", end='\t')
#first_key = list(THPUTS.keys())[0]
for dataset in datasets:
    print("{:s}".format(dataset), end='\t')
print()

for algo in THPUTS:
    print("{:s}".format(algo), end='\t')
    for dataset in datasets:
        if(dataset in THPUTS[algo]):
            if isinstance(THPUTS[algo][dataset], list):
                print("{:.2f}".format(sum(THPUTS[algo][dataset]) / len(THPUTS[algo][dataset]) / 1000), end='\t')
            else:
                print("{:.2f}".format(THPUTS[algo][dataset] / 1000), end='\t')
        else:
            print("NA", end='\t')
    print()
print()
'''