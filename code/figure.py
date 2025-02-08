from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
import matplotlib as mpl
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
mpl.rcParams['text.usetex'] = True
fig = plt.figure()

ax = fig.add_subplot(111,projection = '3d')

# Make data.

data_test = pd.read_csv(r"D:\pycharmproject\second try\400-800.csv", header=None)
x = np.array(data_test.values)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_real = torch.from_numpy(x)
data_we =pd.read_csv(r"D:\pycharmproject\second try\we400-800.csv", header=None)
we = np.array(data_we.values)
we = torch.from_numpy(we)
obs = torch.cat((we, x_real), 1)
'''sequence = list(range(0, 2000))
t =torch.tensor(sequence,dtype=float)
times = t/10000
times = times.unsqueeze(1)
obs = torch.cat((times, x_real), 1)'''
obsc = obs.cpu()
z = np.array([o.detach().numpy() for o in obsc])
z = np.reshape(z, [-1, 3])
X = z[:, 0]
Y = z[:, 1]
Z = z[:, 2]
# Plot the surface.
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 12,
        }
ax.set_xlabel('$\omega_m$[rpm]',fontdict=font)
ax.set_ylabel('$i_d$[A]',fontdict=font)
ax.set_zlabel('$i_q$[A]',fontdict=font)
plt.rc('font',family='Times New Roman')

surf = ax.plot_trisurf(X,Y,Z, cmap=cm.coolwarm,

                       linewidth=0, antialiased=False)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# Customize the z axis.

ax.set_zlim(-1, 1)


ax.zaxis.set_major_locator(LinearLocator(5))

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.

cbr = fig.colorbar(surf, shrink=0.5, aspect=5,pad = 0.2)
cbr.set_ticks([0.2, 0.4, 0.6, 0.8])
plt.savefig(r"target.svg", format="svg")
plt.show()