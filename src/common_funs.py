import math

def progress_bar( progress, data_len, bar_max_len = 30):
	bar_len = math.ceil( ( progress / data_len ) * bar_max_len )
	bar = "#" * bar_len
	bar_filler = " "*( bar_max_len - bar_len )
	print("[" + bar + bar_filler + "]", end="\r")