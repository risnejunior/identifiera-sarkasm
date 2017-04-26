import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
Requires matplot lib
"""


filename = 'predictions.pickle'
directory = 'logs'
file_path = os.path.join('.', directory, filename)
with open(file_path, 'rb') as handle:
	saved = pickle.load(handle)

sample_count = len(saved)
tally = saved.tallyPredictions()

print(tally)
print(sample_count)

plt.axis([0, 30, 0, sample_count])
plt.ion()

for point in tally:
    plt.scatter(point[0][0], point[1])
    plt.pause(0.05)

while True:
    plt.pause(0.05)