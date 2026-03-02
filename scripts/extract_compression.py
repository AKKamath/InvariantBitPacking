import re
import os
import sys
import numpy as np
import math
import glob

RATIOS = {}
THPUTS = {}
COMP_TIME = {}
PREPROC_TIME = {}
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
            matches = re.findall(r'(.*): Time taken to compress ([0-9]+\.[0-9]+) ms. Time taken to decompress: ([0-9]+\.[0-9]+) ms. Throughput: ([0-9]+\.[0-9]+) MB/s', line)
            matches2 = re.findall(r'(.*): Time taken to decompress: ([0-9]+\.[0-9]+) ms. Throughput: ([0-9]+\.[0-9]+) MB/s', line)
            match_preproc = re.findall(r'Finished compression preprocessing; compressed len [0-9]+; orig [0-9]+; time ([0-9]+\.[0-9]+) ms', line)
            layer_num = re.findall(r'Layer ([0-9]+)', line)
            if matches2 and not matches:
                matches = [(match[0], '0.0', match[1], match[2]) for match in matches2]
            if len(layer_num) > 0:
                layer = int(layer_num[0])
            for match in matches:
                if match[0] not in THPUTS:
                    THPUTS[match[0]] = {}
                if match[0] not in COMP_TIME:
                    COMP_TIME[match[0]] = {}
                thput = float(match[3])
                comp_time = float(match[1])
                THPUTS[match[0]][dataset] = THPUTS[match[0]].get(dataset, []) + [thput]
                COMP_TIME[match[0]][dataset] = COMP_TIME[match[0]].get(dataset, []) + [comp_time]
                if layer != -1:
                    THPUTS[match[0]][dataset + "_min"] = min(THPUTS[match[0]].get(dataset + "_min", math.inf), thput)
                    THPUTS[match[0]][dataset + "_max"] = max(THPUTS[match[0]].get(dataset + "_max", -1), thput)
                    COMP_TIME[match[0]][dataset + "_min"] = min(COMP_TIME[match[0]].get(dataset + "_min", math.inf), comp_time)
                    COMP_TIME[match[0]][dataset + "_max"] = max(COMP_TIME[match[0]].get(dataset + "_max", -1), comp_time)

            if match_preproc:
                if dataset not in PREPROC_TIME:
                    PREPROC_TIME[dataset] = []
                preproc_time = float(match_preproc[0])
                PREPROC_TIME[dataset] += [preproc_time]
                if layer != -1:
                    PREPROC_TIME[dataset + "_min"] = min(PREPROC_TIME.get(dataset + "_min", math.inf), preproc_time)
                    PREPROC_TIME[dataset + "_max"] = max(PREPROC_TIME.get(dataset + "_max", -1), preproc_time)

# Example usage
folder_path = sys.argv[1]
datasets = sys.argv[2].split()

for dataset in datasets:
    extract_ratio(folder_path + '/' + dataset + '.log', folder_path)
    extract_thput(folder_path + '/' + dataset + '.log', folder_path)

print("Avg speedup", end='\t')
first_key = list(THPUTS.keys())[0]
datasets = THPUTS[first_key].keys()
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

print("Compression time", end='\t')
first_key = list(COMP_TIME.keys())[0]
for dataset in datasets:
    print("{:s}".format(dataset), end='\t')
print()

for algo in COMP_TIME:
    print("{:s}".format(algo), end='\t')
    for dataset in datasets:
        if(dataset in COMP_TIME[algo]):
            if isinstance(COMP_TIME[algo][dataset], list):
                comp_time = sum(COMP_TIME[algo][dataset]) / len(COMP_TIME[algo][dataset])
            else:
                comp_time = COMP_TIME[algo][dataset]
            print("{:.2f}".format(comp_time), end='\t')
        else:
            print("NA", end='\t')
    print()
print()

print("Preprocessing time", end='\t')
first_key = list(PREPROC_TIME.keys())[0]
for dataset in datasets:
    print("{:s}".format(dataset), end='\t')
print()

print("", end='\t')
for dataset in datasets:
    if(dataset in PREPROC_TIME):
        if isinstance(PREPROC_TIME[dataset], list):
            preproc_time = sum(PREPROC_TIME[dataset]) / len(PREPROC_TIME[dataset])
        else:
            preproc_time = PREPROC_TIME[dataset]
    else:
        preproc_time = 0.0
    print("{:.2f}".format(preproc_time), end='\t')
print()

print("% preprocessing time", end='\t')
first_key = list(PREPROC_TIME.keys())[0]
for dataset in datasets:
    print("{:s}".format(dataset), end='\t')
print()

print("", end='\t')
for dataset in datasets:
    if(dataset in PREPROC_TIME):
        if isinstance(PREPROC_TIME[dataset], list):
            preproc_time = sum(PREPROC_TIME[dataset]) / len(PREPROC_TIME[dataset])
        else:
            preproc_time = PREPROC_TIME[dataset]
    else:
        preproc_time = 0.0

    if isinstance(COMP_TIME["IBP"][dataset], list):
        comp_time = sum(COMP_TIME["IBP"][dataset]) / len(COMP_TIME["IBP"][dataset])
    else:
        comp_time = COMP_TIME["IBP"][dataset]
    preproc_time = preproc_time / comp_time * 100.0 if dataset in COMP_TIME["IBP"] else 0.0
    print("{:.2f}".format(preproc_time), end='\t')
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