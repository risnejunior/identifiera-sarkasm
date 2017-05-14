import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
Requires matplot lib
"""

def test():
	print('test')

options = {
	1: ("Show histogram", test),
	2: ("show similarity", test)
	
}

for k, (text, fun) in options.items():
	print("{}: {}".format(k, text))

while True:
	sel = input("Select sctipt to run:")
	try:
		sel = int(sel)
		if sel > len(options) or sel < 1:
			raise Exception()
	except:
		pass
	else:
		break 

# run method selected
options[sel][1]()
quit()


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