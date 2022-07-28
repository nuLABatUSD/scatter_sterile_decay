import numpy as np

f_TINY = 1e-20
f_MINI = 1e-25
f_SMALL = 1e-30
f_BUFFER = 1e-40
MIN_eps_BUFFER = 12
a_MAXMULT = 2

eps_small_MAX = 1000
len_y_MAX = 2000

small_boxsize = 0.5
eps_small_box = 16
num_small_boxes = int(np.round(eps_small_box/small_boxsize,0)) + 1
initial_boxsize = 0.5
