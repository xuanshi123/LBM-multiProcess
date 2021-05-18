# from PySide2 import QtWidgets
# from ui import main
# import sys
# import os
import math
import numpy as np
from mesh_vox import read_and_reshape_stl, voxelize
# from LBM import preprocessing
from LBM import lbm
import multiprocessing
from numba import cuda

w = np.array([1.0 / 3,
              1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18,
              1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36,
              1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36])


class MyApp:
    def __init__(self):

        self.model_fileName = ''
        self.output_dir = ''

        self.max_u_p = 1

        self.resolution = 100
        self.outer_height = 10
        self.lb_unit = 0.05

        self.max_u_lb = 0.05

        self.dia_u_p = 0.01
        self.out_u_p = 0.01

        self.interval = 0.01
        self.duration = 1
        self.frequency = 300
        self.outer_flow_direction = 'no flow'
        self.Nu_P = 0.002
        self.dia_delay = 0

    def voxelize(self):

        resolution = int(self.resolution)
        mesh, self.bounding_box = read_and_reshape_stl(self.model_fileName, resolution)

        process_number = int(multiprocessing.cpu_count()) * 2
        self.voxels = voxelize(mesh, self.bounding_box, process_number)

    def generate_final_model(self):

        h = round(self.outer_height / self.lb_unit)

        self.voxels = np.delete(self.voxels, np.s_[self.bounding_box[1]:], 0)

        outer = np.zeros((h, self.bounding_box[0], self.bounding_box[2]), dtype=np.int8)

        if self.outer_flow_direction == 'no flow':
            outer[:, 0, :] = 1
            outer[:, -1, :] = 1
        elif self.outer_flow_direction == 'left to right':
            outer[:, 0, :] = 3
            outer[:, -1, :] = 1

        outer[-1, :, :] = 1

        outer[:, :, 0] = 1

        outer[:, :, -1] = 1

        for i in range(self.bounding_box[0]):

            ff = 1
            bf = 1

            ffl = 1
            bfl = 1

            for j in range(self.bounding_box[2]):

                if ffl == 1 and self.voxels[0, i, j] > -1:

                    self.voxels[0, i, j] = -1

                elif self.voxels[0, i, j] < 0:
                    ffl = 0

                if bfl == 1 and self.voxels[0, i, -1 - j] > -1:

                    self.voxels[0, i, -1 - j] = -1

                elif self.voxels[0, i, -1 - j] < 0:
                    bfl = 0

        x = self.voxels[0]

        x[x == 0] = 2

        self.voxels[0] = x

        self.voxels = np.append(self.voxels, outer, axis=0)

    # rotate model around x axis - direction
    def rotate_x_minus(self):

        self.voxels = np.flip(np.transpose(self.voxels, (2, 1, 0)), 0)

        bb0 = self.bounding_box[0]
        bb1 = self.bounding_box[1]
        bb2 = self.bounding_box[2]

        self.bounding_box[1] = bb2
        self.bounding_box[2] = bb1

    # rotate model around x axis + direction
    def rotate_x_plus(self):

        self.voxels = np.transpose(np.flip(self.voxels, 0), (2, 1, 0))

        bb0 = self.bounding_box[0]
        bb1 = self.bounding_box[1]
        bb2 = self.bounding_box[2]

        self.bounding_box[1] = bb2
        self.bounding_box[2] = bb1

    # rotate model around z axis - direction
    def rotate_z_minus(self):

        self.voxels = np.transpose(np.flip(self.voxels, 1), (0, 2, 1))

        bb0 = self.bounding_box[0]
        bb1 = self.bounding_box[1]
        bb2 = self.bounding_box[2]

        self.bounding_box[0] = bb2
        self.bounding_box[2] = bb0

    # rotate model around z axis + direction
    def rotate_z_plus(self):

        self.voxels = np.flip(np.transpose(self.voxels, (0, 2, 1)), 1)

        bb0 = self.bounding_box[0]
        bb1 = self.bounding_box[1]
        bb2 = self.bounding_box[2]

        self.bounding_box[0] = bb2
        self.bounding_box[2] = bb0

    # run simulation         set output folder as parameter
    def start_simulation(self, out_folder='result'):

        lu = self.lb_unit

        duration = self.duration
        max_u_lb = self.max_u_lb
        dia_u_p = self.dia_u_p
        out_u_p = self.out_u_p

        frequency = self.frequency

        interval = self.interval

        ts = lu * max_u_lb / abs(self.max_u_p)

        if abs(out_u_p) / (lu / ts) > 0.2:
            print('External Flow Speed Exceed 0.2 u')

        Nulb = lu * lu / ts

        dia_u_lb = dia_u_p * ts / lu

        if self.outer_flow_direction == 'no flow':
            out_u_lb = 0
        else:
            out_u_lb = out_u_p * ts / lu

        print("Time Step: {} s".format(ts))

        tau_inv = 1.0 / ((self.Nu_P / Nulb) * 3.0 + 0.5)

        print("tau_inverse: {}".format(tau_inv))

        # self.voxels = np.ascontiguousarray(np.transpose(self.voxels, (1, 0, 2)))

        self.voxels = np.zeros((3000, 1000, 3), dtype=np.int8)

        self.voxels[:800, :500, :] = -1

        self.voxels[:, 0, :] = -1

        if self.outer_flow_direction == 'no flow':
            ext_dir = 0
        elif self.outer_flow_direction == 'left to right':
            ext_dir = 3

        lbm.solve(duration, lu, dia_u_lb, out_u_lb, tau_inv, ts, self.voxels, ext_dir, interval, frequency,
                  self.dia_delay, out_folder)

    # run simulation         set output folder as parameter
    def load_file(self, file_name, time, out_folder='result'):

        lu = self.lb_unit

        duration = self.duration
        max_u_lb = self.max_u_lb
        dia_u_p = self.dia_u_p
        out_u_p = self.out_u_p

        frequency = self.frequency

        interval = self.interval

        ts = lu * max_u_lb / abs(self.max_u_p)

        if abs(out_u_p) / (lu / ts) > 0.2:
            print('External Flow Speed Exceed 0.2 u')

        Nulb = lu * lu / ts

        dia_u_lb = dia_u_p * ts / lu

        if self.outer_flow_direction == 'no flow':
            out_u_lb = 0
        else:
            out_u_lb = out_u_p * ts / lu

        print("Time Step: {} s".format(ts))

        tau_inv = 1.0 / ((self.Nu_P / Nulb) * 3.0 + 0.5)

        print("tau_inverse: {}".format(tau_inv))

        # self.voxels = np.ascontiguousarray(np.transpose(self.voxels, (1, 0, 2)))

        ftable = np.load(file_name)

        lbm.conti(duration, time, lu, dia_u_lb, out_u_lb, tau_inv, ts, ftable, interval, frequency,
                  self.dia_delay, out_folder)

    # run simulation         set output folder as parameter
    def load_file_2(self, file_name, time, out_folder='result'):

        lu = self.lb_unit

        duration = self.duration
        max_u_lb = self.max_u_lb
        dia_u_p = self.dia_u_p
        out_u_p = self.out_u_p

        frequency = self.frequency

        interval = self.interval

        ts = lu * max_u_lb / abs(self.max_u_p)

        if abs(out_u_p) / (lu / ts) > 0.2:
            print('External Flow Speed Exceed 0.2 u')

        Nulb = lu * lu / ts

        dia_u_lb = dia_u_p * ts / lu

        if self.outer_flow_direction == 'no flow':
            out_u_lb = 0
        else:
            out_u_lb = out_u_p * ts / lu

        print("Time Step: {} s".format(ts))

        tau_inv = 1.0 / ((self.Nu_P / Nulb) * 3.0 + 0.5)

        print("tau_inverse: {}".format(tau_inv))

        self.voxels = np.ascontiguousarray(np.transpose(self.voxels[1:, :, 1:], (1, 0, 2)))

        print(self.voxels.shape)

        threadsperblock = (16, 16)

        ftable_2d = np.load(file_name)

        rho_nozzle = np.sum(ftable_2d[400, 500])

        rho = np.sum(ftable_2d, axis=2)

        norm = np.amin(rho[rho != 0])

        for k in range(19):
            ftable_2d[:, :, k] = np.round(ftable_2d[:, :, k] / norm / w[k] * 30000)

        print(np.max(ftable_2d))

        ftable_2d = ftable_2d.astype(np.uint16)

        ftable_2d = np.ascontiguousarray(np.flip(ftable_2d, 1))

        ftable_2d = np.reshape(ftable_2d, (ftable_2d.shape[0], ftable_2d.shape[1], 1, 19))
        ftable = np.repeat(ftable_2d, 400, axis=2)

        blockspergrid_y = math.ceil(self.voxels.shape[1] / threadsperblock[0])
        blockspergrid_z = math.ceil(self.voxels.shape[2] / threadsperblock[1])
        blockspergrid = (blockspergrid_y, blockspergrid_z)

        print(rho_nozzle)

        lbm.initialization(self.voxels, ftable[200:], blockspergrid, threadsperblock, 1)

        ftable[200: 600, 500:600] = ftable[200: 600, 701:801]
        ftable[200: 600, 600:] = 0

        lbm.conti(duration, time, lu, dia_u_lb, out_u_lb, tau_inv, ts, ftable, interval, frequency,
                  self.dia_delay, out_folder)


if __name__ == '__main__':

    # create simulator object
    app = MyApp()

    # num of lattices along the longest dimension of the nozzle model
    app.resolution = 501
    # dx between lattice nodes in m
    app.lb_unit = 0.02 / 200
    # height of external region in m
    app.outer_height = 20 / 1000

    # maximum physical diaphram velocity
    app.dia_u_p = 0.07
    # maximum physical side flow velocity
    app.out_u_p = 0.5
    # duration of simulation in s
    app.duration = 105 / 1000
    # diaphragm oscillation frequency
    app.frequency = 40

    # data output interval
    app.interval = 5 / 1000
    # max LB velocity
    app.max_u_lb = 0.2

    app.max_u_p = 1.5
    # physical k viscosity
    app.Nu_P = 15 / 1000000
    # nozzle model file name
    app.model_fileName = 'sja.STL'

    app.outer_flow_direction = 'left to right'

    app.dia_delay = 2.7800000000103813

    # turn STL model into an array of Lattice Domain

    app.voxelize()

    with open('resulttime.dat', 'r') as f:
        t = float(f.readline())	

    # t = 2.7800000000103813
    # app.load_file_2('ftable_origin.npy', t, 'result1')

    app.load_file('resultftable.npy', t, 'result')
