from pyedflib import highlevel
import numpy as np
import struct
import math
import pickle

folder = "./dataset/"
# Using readlines()
file1 = open(folder+'RECORDS', 'r')
Lines = file1.readlines()
files = []

# Baseline, eyes open
# Baseline, eyes closed
# Task 1 (open and close left or right fist)
# Task 2 (imagine opening and closing left or right fist)
# Task 3 (open and close both fists or both feet)
# Task 4 (imagine opening and closing both fists or both feet)
# Task 1
# Task 2
# Task 3
# Task 4
# Task 1
# Task 2
# Task 3
# Task 4

for i in range(14):
    files.append([])

# Strips the newline character
for line in Lines:
    temp = line.strip()
    temp = int(temp[-6:-4])
    files[temp-1].append(folder+line.strip())

# for i in range(14):
#     del files[i][87]  # his annotations suck
#     del files[i][90]  # his annotations suck

# print(len(files[0]))

num_part = 109

set1 = [0, 1, 2, 3, 6, 7, 10, 11]
set2 = [4, 5, 8, 9, 12, 13]

count = 0

data_label = []
data_value = []
for j in range(num_part):
    data_x_p = []
    data_y_p = []
    for i in range(14):
        signals, signal_headers, header = highlevel.read_edf(files[i][j])
        sam_rate = signal_headers[0]['sample_rate']
        data_x = []
        data_y = []
        for k in header['annotations']:
            if k[0] != 0:
                start = int(k[0]*sam_rate)-1
            else:
                start = 0
            end = int(float(k[1].decode())*sam_rate)
            if (start+end < len(signals[0])):
                val = [item for item in range(start, start+end)]
            else:
                count += 1
                val = [item for item in range(start, len(signals[0]))]
            if (i in set1):
                if(k[2] == 'T0'):
                    y_hat = 0  # rest
                elif(k[2] == 'T1'):
                    y_hat = 1  # left
                elif(k[2] == 'T2'):
                    y_hat = 2  # right
            else:
                if(k[2] == 'T0'):
                    y_hat = 0  # rest
                elif(k[2] == 'T1'):
                    y_hat = 3  # both fist
                elif(k[2] == 'T2'):
                    y_hat = 4  # both feet
            input_data = signals.T[val]

            data_x.append(input_data)  # unprocessed data
            data_y.append(y_hat)
        print(i, j)
        data_x_p.append(data_x)
        data_y_p.append(data_y)
    save = []
    save.append(data_x_p)
    save.append(data_y_p)
    with open(folder+"data_txt/data_"+str(j)+'.txt', 'wb') as F:
        # Dump the list to file
        pickle.dump(save, F)

    F.close()
