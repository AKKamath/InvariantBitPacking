import re
import os
import sys
import numpy as np
import math
import glob

RATIOS = {}
THPUTS = {}
layer = 0
def extract_ratio(file_path, folder_path):
    global layer
    dataset = re.findall(folder_path + '/(.*).log', file_path)[0]
    with open(file_path, 'r') as file:
        for line in file:
            layer_m = re.findall(r'Layer (.*)', line)
            if len(layer_m) != 0:
                layer = int(layer_m[0])
                if layer not in RATIOS:
                    RATIOS[layer] = {}
            matches = re.findall(r'(.*): Uncompressed bytes: [0-9]+, compressed bytes: [0-9]+, ratio: ([0-9]+\.[0-9]+)', line)
            for match in matches:
                if(match[0] != "Us"):
                    continue
                print(dataset, layer, match[1])
                RATIOS[layer][dataset] = RATIOS[layer].get(dataset, []) + [float(match[1])]

def extract_thput(file_path, folder_path):
    dataset = re.findall(folder_path + '/(.*).log', file_path)[0]
    with open(file_path, 'r') as file:
        content = file.read()
        matches = re.findall(r'(.*): Time taken to decompress: ([0-9]+\.[0-9]+) ms. Throughput: ([0-9]+\.[0-9]+) MB/s', content)
        for match in matches:
            if match[0] not in THPUTS:
                THPUTS[match[0]] = {}
            THPUTS[match[0]][dataset] = THPUTS[match[0]].get(dataset, []) + [float(match[2])]

# Example usage
folder_path = sys.argv[1]
datasets = sys.argv[2].split()

for dataset in datasets:
    extract_ratio(folder_path + '/' + dataset + '.log', folder_path)
    extract_thput(folder_path + '/' + dataset + '.log', folder_path)


print("Layers", end='\t')
#first_key = list(RATIOS.keys())[0]
#datasets = RATIOS[first_key]
for dataset in datasets:
    print("{:s}".format(dataset), end='\t')
print()

for layer in RATIOS:
    print("{:d}".format(layer), end='\t')
    for dataset in datasets:
        if(dataset in RATIOS[layer]):
            print("{:.2f}".format(sum(RATIOS[layer][dataset]) / len(RATIOS[layer][dataset])), end='\t')
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
            print("{:.2f}".format(sum(THPUTS[algo][dataset]) / len(THPUTS[algo][dataset]) / 1000), end='\t')
        else:
            print("NA", end='\t')
    print()
print()

print("Norm. thput", end='\t')
first_key = list(THPUTS.keys())[0]
for dataset in datasets:
    print("{:s}".format(dataset), end='\t')
print()

for algo in THPUTS:
    print("{:s}".format(algo), end='\t')
    for dataset in datasets:
        if(dataset in THPUTS[algo] and dataset in THPUTS["Transfer"]):
            transfer = sum(THPUTS["Transfer"][dataset]) / len(THPUTS["Transfer"][dataset])
            thput = sum(THPUTS[algo][dataset]) / len(THPUTS[algo][dataset])
            print("{:.2f}".format(thput / transfer), end='\t')
        else:
            print("NA", end='\t')
    print()
'''