import os
import re
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':

    path = '.'
    dirFiles = [f for f in os.listdir(path) if f.endswith('.npy')]
    dirFiles.sort(key = lambda s: float(re.search('\d+(\.\d+)?', s).group()))

    print(dirFiles)
    for x in dirFiles:
        f = np.load(x)
        rho = f[2]
        dpi_i = 100 if np.max(rho.shape) / 3 < 100 else round(np.max(rho.shape) / 150) * 50
        t = float(re.search('\d+(\.\d+)?', x).group())
        print(t)
        plt.clf()
        plt.title('rho')
        plt.xlabel('x ')
        plt.ylabel('y ')
        plt.imshow(rho, cmap=cm.plasma)
        plt.colorbar()
        plt.gca().invert_yaxis()
        fname = "Vy time {:11.9f} ms.png".format(t)
        plt.savefig(fname, dpi=dpi_i, bbox_inches='tight', pad_inches=0)