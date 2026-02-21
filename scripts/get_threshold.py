import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

perc_list = {}
def extract_thput_base(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    for line in data:
        # Regex to match your data format
        match = re.match(r"Threshold ([0-9.]+): Saved bits per element: [0-9]+ \(Total [0-9]+, ([0-9.]+)%\)", line.strip())
        if match:
            thresh = float(match.group(1))
            perc = float(match.group(2))
            if thresh not in perc_list:
                perc_list[thresh] = []
            perc_list[thresh].append(perc)

def main():
    file = sys.argv[1]
    extract_thput_base(file)

    sizes = sys.argv[2].split()
    print("Threshold", end="\t")
    for i in sizes:
        print(i + "B", end="\t")
    print()
    for i in perc_list.keys():
        print(str(int(i * 100)) + "%", end="\t")
        for j in perc_list[i]:
            print(f"{j:.1f}%", end="\t")
        print()


    

if __name__ == "__main__":
    main()
