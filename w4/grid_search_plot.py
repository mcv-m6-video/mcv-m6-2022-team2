
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
from numpy.random import randn
from scipy import array, newaxis
import pickle

with open('./variables/TM_CCOEFF_NORMED/pepn.pickle', 'rb') as f:
    msen = pickle.load(f)
    
window_sizes = [2,4,8,16,32,64]
search_areas = [2,4,8,16,32,64]

list_res = []
counter = 0
for i in range(len(window_sizes)):
    for j in range(len(search_areas)):
        list_res.append({'window':window_sizes[i], 'search_area':search_areas[j],'MSEN': msen[counter]})
        counter += 1


""" list_res = [{'alpha': 3, 'rho': 0.01, 'AP': 0.3867}, {'alpha': 3, 'rho': 0.03, 'AP': 0.3872}, {'alpha': 3, 'rho': 0.049999999999999996, 'AP': 0.365}, {'alpha': 3, 'rho': 0.06999999999999999, 'AP': 0.338}, {'alpha': 3, 'rho': 0.08999999999999998, 'AP': 0.3175}, {'alpha': 4, 'rho': 0.01, 'AP': 0.4083}, {'alpha': 4, 'rho': 0.03, 'AP': 0.4029}, {'alpha': 4, 'rho': 0.049999999999999996, 'AP': 0.3995}, {'alpha': 4, 'rho': 0.06999999999999999, 'AP': 0.3131}, {'alpha': 4, 'rho': 0.08999999999999998, 'AP': 0.3125}, {'alpha': 5, 'rho': 0.01, 'AP': 0.2993}, {'alpha': 5, 'rho': 0.03, 'AP': 0.2291}, {'alpha': 5, 'rho': 0.049999999999999996, 'AP': 0.2159}, {'alpha': 5, 'rho': 0.06999999999999999, 'AP': 0.2001}, {'alpha': 5, 'rho': 0.08999999999999998, 'AP': 0.1971}, {'alpha': 6, 'rho': 0.01, 'AP': 0.1463}, {'alpha': 6, 'rho': 0.03, 'AP': 0.1388}, {'alpha': 6, 'rho': 0.049999999999999996, 'AP': 0.1345}, {'alpha': 6, 'rho': 0.06999999999999999, 'AP': 0.1351}, {'alpha': 6, 'rho': 0.08999999999999998, 'AP': 0.132}, {'alpha': 7, 'rho': 0.01, 'AP': 0.1357}, {'alpha': 7, 'rho': 0.03, 'AP': 0.1316}, {'alpha': 7, 'rho': 0.049999999999999996, 'AP': 0.0909}, {'alpha': 7, 'rho': 0.06999999999999999, 'AP': 0.0909}, {'alpha': 7, 'rho': 0.08999999999999998, 'AP': 0.0909}]
print(list_res) """

x = []
y = []
z = []
for dict_res in list_res:
    print(dict_res)
    x.append(dict_res['window'])
    y.append(dict_res['search_area'])
    z.append(dict_res['MSEN'])
print(x, y, z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Grid Search MSEN')
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
ax.set_xlabel('Window')
ax.set_zlabel('MSEN')
ax.set_ylabel('Search Area')
fig.tight_layout()

plt.show()

