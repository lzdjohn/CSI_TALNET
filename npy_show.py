# import numpy as np

# npy = np.load("./csi_signal/signal/1_sjz_0.npy")
# print(npy[0][0])
# print(npy[0][0].real)
# print(npy[0][0].imag)
# for i in range(len(npy[0])):
#     if npy[0][i].imag != 0:
#         print(npy[0][i].imag)
# print("done")

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from bayesian_changepoint_detection.generate_data import generate_normal_time_series


npy = np.load("./csi_signal/signal/1_sjz_0.npy")
partition, data = generate_normal_time_series(7, 50, 200)
fig, ax = plt.subplots(figsize=[16, 12])
ax.plot(data)
