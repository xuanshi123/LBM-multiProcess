import numpy as np
import os as os
from matplotlib import cm
import matplotlib.pyplot as plt
import math
# from evtk.hl import pointsToVTK

w = np.array([1.0 / 3,
              1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18,
              1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36,
              1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36])


def output_result(lu, ts, ftemp, result_folder, t):

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
        print("Result Directory ", result_folder, " Created ")

    ftemp = ftemp.astype(np.float32)

    for k in range(19):
        ftemp[:, :, k] = ftemp[:, :, k] / 30000 * w[k]

    rho = np.sum(ftemp, axis=2)

    dpi_i = 100 if np.max(rho.shape) / 3 < 100 else round(np.max(rho.shape) / 150) * 50

    plt.clf()
    plt.title('rho')
    plt.xlabel('x ')
    plt.ylabel('y ')
    plt.imshow(rho, cmap=cm.plasma)
    plt.colorbar()
    plt.gca().invert_yaxis()
    fname = result_folder + "/rho {:11.9f} ms.png".format(t * 1000)
    plt.savefig(fname, dpi=dpi_i, bbox_inches='tight', pad_inches=0)

    vx = ftemp[:, :, 1] - ftemp[:, :, 2] + ftemp[:, :, 7] - ftemp[:, :, 8] + ftemp[:, :, 10] - ftemp[:, :, 9] + \
         ftemp[:, :, 11] - ftemp[:, :, 12] + ftemp[:, :, 16] - ftemp[:, :, 15]

    vy = ftemp[:, :, 3] - ftemp[:, :, 4] + ftemp[:, :, 7] - ftemp[:, :, 8] + ftemp[:, :, 9] - ftemp[:, :, 10] + \
         ftemp[:, :, 13] - ftemp[:, :, 14] + ftemp[:, :, 18] - ftemp[:, :, 17]

    # vz = ftemp[:, :, 5] - ftemp[:, :, 6] + ftemp[:, :, 11] - ftemp[:, :, 12] + ftemp[:, :, 13] - ftemp[:, :, 14] + \
    #      ftemp[:, :, 15] - ftemp[:, :, 16] + ftemp[:, :, 17] - ftemp[:, :, 18]

    rho[rho == 0] = np.amin(rho[rho != 0])

    vx_p = vx / rho * lu / ts

    vy_p = vy / rho * lu / ts

    # vz_p = vz / rho * lu / ts


    plt.clf()
    plt.title('Vx m/s')
    plt.xlabel('x ')
    plt.ylabel('y ')
    cap = np.max(np.abs(vx_p))
    plt.imshow(vx_p, cmap=cm.plasma, vmin=-cap, vmax=cap)
    plt.colorbar()
    plt.gca().invert_yaxis()
    fname = result_folder + "/Vx time {:11.9f} ms.png".format(t * 1000)
    plt.savefig(fname, dpi=dpi_i, bbox_inches='tight', pad_inches=0)

    plt.clf()
    plt.title('Vy m/s')
    plt.xlabel('x ')
    plt.ylabel('y ')
    cap = np.max(np.abs(vy_p))
    plt.imshow(vy_p, cmap=cm.plasma, vmin=-cap, vmax=cap)
    plt.colorbar()
    plt.gca().invert_yaxis()
    fname = result_folder + "/Vy time {:11.9f} ms.png".format(t * 1000)
    plt.savefig(fname, dpi=dpi_i, bbox_inches='tight', pad_inches=0)

    # velocity_p = np.sqrt(vx_p * vx_p + vy_p * vy_p)
    # plt.clf()
    # plt.title('total velocity m/s')
    # plt.xlabel('x ')
    # plt.ylabel('y ')
    # plt.imshow(velocity_p, cmap=cm.plasma)
    # plt.colorbar()
    # plt.gca().invert_yaxis()
    # fname = result_folder + "/total velocity time {:11.9f} ms.png".format(t * 1000)
    # plt.savefig(fname, dpi=dpi_i, bbox_inches='tight', pad_inches=0.01)

    out = np.zeros((3, vx.shape[0], vx.shape[1]), np.float32)

    out[0] = vx_p
    out[1] = vy_p
    out[2] = rho
    fname = result_folder + "/data {:11.9f} ms".format(t * 1000)
    np.save(fname, out)


def output_result_2(lu, ts, ftemp, result_folder, t):
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
        print("Result Directory ", result_folder, " Created ")

    ftemp = ftemp.astype(np.float32)

    for k in range(19):
        ftemp[:, :, k] = ftemp[:, :, k] / 30000 * w[k]

    rho = np.sum(ftemp, axis=2)

    vx = ftemp[:, :, 1] - ftemp[:, :, 2] + ftemp[:, :, 7] - ftemp[:, :, 8] + ftemp[:, :, 10] - ftemp[:, :, 9] + \
         ftemp[:, :, 11] - ftemp[:, :, 12] + ftemp[:, :, 16] - ftemp[:, :, 15]

    vy = ftemp[:, :, 3] - ftemp[:, :, 4] + ftemp[:, :, 7] - ftemp[:, :, 8] + ftemp[:, :, 9] - ftemp[:, :, 10] + \
         ftemp[:, :, 13] - ftemp[:, :, 14] + ftemp[:, :, 18] - ftemp[:, :, 17]

    vz = ftemp[:, :, 5] - ftemp[:, :, 6] + ftemp[:, :, 11] - ftemp[:, :, 12] + ftemp[:, :, 13] - ftemp[:, :, 14] + \
         ftemp[:, :, 15] - ftemp[:, :, 16] + ftemp[:, :, 17] - ftemp[:, :, 18]

    rho[rho == 0] = np.amin(rho[rho != 0])

    vx_p = vx / rho * lu / ts

    vy_p = vy / rho * lu / ts

    vz_p = vz / rho * lu / ts

    out = np.zeros((4, vx.shape[0], vx.shape[1]), np.float32)

    out[0] = vx_p
    out[1] = vy_p
    out[2] = vz_p
    out[3] = rho

    fname = result_folder + "/data {:11.9f} ms".format(t * 1000)
    np.save(fname, out)

