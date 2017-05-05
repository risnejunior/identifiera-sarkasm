import time
import math
import os
import sys

""" 
Cross platform, plays a beeping sound
Useful as a signal when a script or makefile is done running
"""

i = 1.0
switch = False
beeper = None
coma = 1

if os.name == 'nt':
	import winsound
	beeper = lambda freq,dur: winsound.Beep(freq,dur)
	coma = 0
else:
	beeper = lambda freq,dur: print("\a", end="\r")

while True:
	freq = math.floor(500 * i) # Set Frequency
	dur = 400 # Set Duration
	beeper(freq, dur)
	time.sleep(coma)
	
	if i > 1.1:
		switch = True
	elif i < 0.9:
		switch = False

	if switch:
		i -= 0.1 
	else:
		i += 0.1
