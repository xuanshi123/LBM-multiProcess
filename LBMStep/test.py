import numpy as np
import SharedArray as sa
from numba import cuda
from numba import float32
import concurrent.futures as cf
import multiprocessing
from LBM import out_result
import math

e = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                  [0, 0, 1], [0, 0, -1], [1, 1, 0], [-1, -1, 0], [-1, 1, 0],
                  [1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, -1, -1],
                  [-1, 0, 1], [1, 0, -1], [0, -1, 1], [0, 1, -1]], dtype=np.int8)

w = np.array([1.0 / 3,
              1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18,
              1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36,
              1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36], dtype=np.float32)

opp = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int8)

opp2 = np.array([0, 1, 2, 4, 3, 5, 6, 10, 9, 8, 7, 11, 12, 17, 18, 15, 16, 13, 14], dtype=np.int8)


def propagate_start(ft1, ft2, blockspergrid, threadsperblock, gpu_id, end, out_u_lb, tau_inv, state, event, cond, device_num):

    cuda.select_device(gpu_id)

    # thread function to handle streaming step
    @cuda.jit('void(uint16[:,:,:], uint16[:,:,:], uint16[:,:,:], uint16[:,:,:], float32)')
    def propagate_kernel_new(ftemp, dcur, dabove, dbelow, tau_inv):

        # set constant array of direction vectors
        de = cuda.const.array_like(e)
        # set constant array of opposite direction indices
        dopp = cuda.const.array_like(opp)

        dopp2 = cuda.const.array_like(opp2)
        # obtain position in grid
        dw = cuda.const.array_like(w)

        x, y = cuda.grid(2)

        # if within domain and  not solid node
        if x < dcur.shape[0] and y < dcur.shape[1]:

            ftemp[x, y, 0] = dcur[x, y, 0]

            for i in range(1, 19):
                # obtain original position of f distribution
                xi = x + de[i, 1]
                yi = y - de[i, 2]

                if yi == -1:
                    yi = dcur.shape[1] - 1
                elif yi == dcur.shape[1]:
                    yi = 0

                if x == dcur.shape[0] - 1:

                    # if from current layer
                    if de[i, 0] == 0:

                        if xi == dcur.shape[0]:
                            j = dopp[i]
                            ftemp[x, y, i] = dcur[x, y, j]
                        else:
                            ftemp[x, y, i] = dcur[xi, yi, i]

                    # if from layer below
                    elif de[i, 0] > 0:

                        if xi == dcur.shape[0]:
                            j = dopp[i]
                            ftemp[x, y, i] = dcur[x, y, j]
                        else:
                            ftemp[x, y, i] = dbelow[xi, yi, i]
                    # if from layer above
                    elif de[i, 0] < 0:

                        if xi == dcur.shape[0]:
                            j = dopp[i]
                            ftemp[x, y, i] = dcur[x, y, j]
                        else:
                            ftemp[x, y, i] = dabove[xi, yi, i]

                elif x == 0:

                    # if from current layer
                    if de[i, 0] == 0:

                        if xi == -1:
                            j = dopp2[i]
                            ftemp[x, y, i] = dcur[x, yi, j]
                        else:
                            ftemp[x, y, i] = dcur[xi, yi, i]

                    # if from layer below
                    elif de[i, 0] > 0:

                        if xi == -1:
                            j = dopp2[i]
                            ftemp[x, y, i] = dbelow[x, yi, j]
                        else:
                            ftemp[x, y, i] = dbelow[xi, yi, i]
                    # if from layer above
                    elif de[i, 0] < 0:

                        if xi == -1:
                            j = dopp2[i]
                            ftemp[x, y, i] = dabove[x, yi, j]
                        else:
                            ftemp[x, y, i] = dabove[xi, yi, i]

                else:

                    # if from current layer
                    if de[i, 0] == 0:
                        ftemp[x, y, i] = dcur[xi, yi, i]
                    # if from layer below
                    elif de[i, 0] > 0:
                        ftemp[x, y, i] = dbelow[xi, yi, i]
                    # if from layer above
                    elif de[i, 0] < 0:
                        ftemp[x, y, i] = dabove[xi, yi, i]

            f = cuda.local.array(19, dtype=float32)

            rho = 0.0

            for i in range(19):
                f[i] = ftemp[x, y, i] / 30000 * dw[i]
                rho += f[i]

            vx = (f[1] - f[2] + f[7] - f[8] + f[10] - f[9] + f[11] - f[12] + f[16] - f[15]) / rho

            vy = (f[3] - f[4] + f[7] - f[8] + f[9] - f[10] + f[13] - f[14] + f[18] - f[17]) / rho

            vz = (f[5] - f[6] + f[11] - f[12] + f[13] - f[14] + f[15] - f[16] + f[17] - f[18]) / rho

            # calculate feq and f after collision

            square = 1.5 * (vx * vx + vy * vy + vz * vz)

            f[0] += (dw[0] * rho * (1 - square) - f[0]) * tau_inv

            feq1 = dw[1] * rho * (1.0 + 3.0 * vx + 4.5 * vx * vx - square)

            f[1] += (feq1 - f[1]) * tau_inv

            f[2] += (feq1 - 6.0 * dw[1] * rho * vx - f[2]) * tau_inv

            feq3 = dw[3] * rho * (1.0 + 3.0 * vy + 4.5 * vy * vy - square)

            f[3] += (feq3 - f[3]) * tau_inv

            f[4] += (feq3 - 6.0 * dw[3] * rho * vy - f[4]) * tau_inv

            feq5 = dw[5] * rho * (1.0 + 3.0 * vz + 4.5 * vz * vz - square)

            f[5] += (feq5 - f[5]) * tau_inv

            f[6] += (feq5 - 6.0 * dw[5] * rho * vz - f[6]) * tau_inv

            sum = vx + vy

            vxy = 2.0 * vx * vy

            vxy2 = vx * vx + vy * vy

            feq7 = dw[7] * rho * (1.0 + 3.0 * sum + 4.5 * (vxy2 + vxy) - square)

            f[7] += (feq7 - f[7]) * tau_inv
            f[8] += (feq7 - 6.0 * dw[7] * rho * sum - f[8]) * tau_inv

            sum = vy - vx

            feq9 = dw[9] * rho * (1.0 + 3.0 * sum + 4.5 * (vxy2 - vxy) - square)

            f[9] += (feq9 - f[9]) * tau_inv
            f[10] += (feq9 - 6.0 * dw[9] * rho * sum - f[10]) * tau_inv

            sum = vx + vz

            vxz = 2.0 * vx * vz

            vxz2 = vx * vx + vz * vz

            feq11 = dw[11] * rho * (1.0 + 3.0 * sum + 4.5 * (vxz2 + vxz) - square)

            f[11] += (feq11 - f[11]) * tau_inv
            f[12] += (feq11 - 6.0 * dw[11] * rho * sum - f[12]) * tau_inv

            sum = vy + vz

            vyz = 2.0 * vy * vz

            vyz2 = vy * vy + vz * vz

            feq13 = dw[13] * rho * (1.0 + 3.0 * sum + 4.5 * (vyz + vyz2) - square)

            f[13] += (feq13 - f[13]) * tau_inv
            f[14] += (feq13 - 6.0 * dw[13] * rho * sum - f[14]) * tau_inv

            sum = vz - vx

            feq15 = dw[15] * rho * (1.0 + 3.0 * sum + 4.5 * (vxz2 - vxz) - square)

            f[15] += (feq15 - f[15]) * tau_inv
            f[16] += (feq15 - 6.0 * dw[15] * rho * sum - f[16]) * tau_inv

            sum = vz - vy

            feq17 = dw[17] * rho * (1.0 + 3.0 * sum + 4.5 * (vyz2 - vyz) - square)

            f[17] += (feq17 - f[17]) * tau_inv
            f[18] += (feq17 - 6.0 * dw[17] * rho * sum - f[18]) * tau_inv

            for i in range(19):
                ftemp[x, y, i] = round(f[i] * 30000 / dw[i])

    @cuda.jit('void(uint16[:,:,:], uint16[:,:,:], uint16[:,:,:], uint16[:,:,:], float32, float32, int32)')
    def propagate_kernel(ftemp, dcur, dabove, dbelow, tau_inv, v, flag):

        # set constant array of direction vectors
        de = cuda.const.array_like(e)
        # set constant array of opposite direction indices
        dopp = cuda.const.array_like(opp)

        dopp2 = cuda.const.array_like(opp2)

        dw = cuda.const.array_like(w)
        # obtain position in grid
        x, y = cuda.grid(2)

        # if within domain and  not solid node
        if x < dcur.shape[0] and y < dcur.shape[1]:

            if dcur[x, y, 0] > 0:

                ftemp[x, y, 0] = dcur[x, y, 0]

                for i in range(1, 19):

                    # obtain original position of f distribution
                    xi = x + de[i, 1]
                    yi = y - de[i, 2]

                    if yi == -1:
                        yi = dcur.shape[1] - 1
                    elif yi == dcur.shape[1]:
                        yi = 0

                    # if from current layer
                    if de[i, 0] == 0:

                        l = dcur[xi, yi, i]
                        if l == 0 or xi == dcur.shape[0]:
                            j = dopp[i]
                            ftemp[x, y, i] = dcur[x, y, j]
                        elif xi == -1:
                            j = dopp2[i]
                            ftemp[x, y, i] = dcur[x, yi, j]
                        else:
                            ftemp[x, y, i] = l

                    # if from layer below
                    elif de[i, 0] > 0:

                        l = dbelow[xi, yi, i]
                        if l == 0 or xi == dcur.shape[0]:
                            j = dopp[i]
                            ftemp[x, y, i] = dcur[x, y, j]
                        elif xi == -1:
                            j = dopp2[i]
                            ftemp[x, y, i] = dbelow[x, yi, j]
                        else:
                            ftemp[x, y, i] = l
                    # if from layer above
                    elif de[i, 0] < 0:

                        l = dabove[xi, yi, i]
                        if l == 0 or xi == dcur.shape[0]:
                            j = dopp[i]
                            ftemp[x, y, i] = dcur[x, y, j]
                        elif xi == -1:
                            j = dopp2[i]
                            ftemp[x, y, i] = dabove[x, yi, j]
                        else:
                            ftemp[x, y, i] = l

                f = cuda.local.array(19, dtype=float32)

                rho = 0.0

                if flag == 0:

                    for i in range(19):
                        f[i] = ftemp[x, y, i] / 30000 * dw[i]
                        rho += f[i]
                elif flag == 2:

                    for i in range(19):
                        f[i] = ftemp[x, y, i] / 30000 * dw[i]
                        rho += f[i]

                    if x == ftemp.shape[0] - 1:

                        ro = 0.0
                        for i in range(19):
                            ro += dcur[x, y, i] / 30000 * dw[i]

                        b = 6 * ro * v / 36.0

                        f[3] += 6 * ro * v / 18.0
                        f[7] += b
                        f[9] += b
                        f[13] += b
                        f[18] += b
                        rho += ro * v

                else:

                    for i in range(19):
                        f[i] = ftemp[x, y, i] / 30000 * dw[i]

                    ru = (f[0] + f[3] + f[4] + f[5] + f[6] + f[13] + f[14] + f[17] + f[18]
                          + 2 * (f[2] + f[8] + f[9] + f[12] + f[15])) / (1 - v) * v

                    f[1] = f[2] + ru / 3

                    diff_3_4 = f[3] - f[4]

                    diff_13_14 = f[13] - f[14]

                    diff_18_17 = f[18] - f[17]

                    diff_5_6 = f[5] - f[6]

                    f[7] = f[8] + ru / 6 - (diff_3_4 + diff_13_14 + diff_18_17) / 2

                    f[10] = f[9] + ru / 6 + (diff_3_4 + diff_13_14 + diff_18_17) / 2

                    f[11] = f[12] + ru / 6 - (diff_5_6 + diff_13_14 - diff_18_17) / 2

                    f[16] = f[15] + ru / 6 + (diff_5_6 + diff_13_14 - diff_18_17) / 2

                    for i in range(19):
                        rho += f[i]

                vx = (f[1] - f[2] + f[7] - f[8] + f[10] - f[9] + f[11] - f[12] + f[16] - f[15]) / rho

                vy = (f[3] - f[4] + f[7] - f[8] + f[9] - f[10] + f[13] - f[14] + f[18] - f[17]) / rho

                vz = (f[5] - f[6] + f[11] - f[12] + f[13] - f[14] + f[15] - f[16] + f[17] - f[18]) / rho

                # calculate feq and f after collision

                square = 1.5 * (vx * vx + vy * vy + vz * vz)

                f[0] += (dw[0] * rho * (1 - square) - f[0]) * tau_inv

                feq1 = dw[1] * rho * (1.0 + 3.0 * vx + 4.5 * vx * vx - square)

                f[1] += (feq1 - f[1]) * tau_inv

                f[2] += (feq1 - 6.0 * dw[1] * rho * vx - f[2]) * tau_inv

                feq3 = dw[3] * rho * (1.0 + 3.0 * vy + 4.5 * vy * vy - square)

                f[3] += (feq3 - f[3]) * tau_inv

                f[4] += (feq3 - 6.0 * dw[3] * rho * vy - f[4]) * tau_inv

                feq5 = dw[5] * rho * (1.0 + 3.0 * vz + 4.5 * vz * vz - square)

                f[5] += (feq5 - f[5]) * tau_inv

                f[6] += (feq5 - 6.0 * dw[5] * rho * vz - f[6]) * tau_inv

                sum = vx + vy

                vxy = 2.0 * vx * vy

                vxy2 = vx * vx + vy * vy

                feq7 = dw[7] * rho * (1.0 + 3.0 * sum + 4.5 * (vxy2 + vxy) - square)

                f[7] += (feq7 - f[7]) * tau_inv
                f[8] += (feq7 - 6.0 * dw[7] * rho * sum - f[8]) * tau_inv

                sum = vy - vx

                feq9 = dw[9] * rho * (1.0 + 3.0 * sum + 4.5 * (vxy2 - vxy) - square)

                f[9] += (feq9 - f[9]) * tau_inv
                f[10] += (feq9 - 6.0 * dw[9] * rho * sum - f[10]) * tau_inv

                sum = vx + vz

                vxz = 2.0 * vx * vz

                vxz2 = vx * vx + vz * vz

                feq11 = dw[11] * rho * (1.0 + 3.0 * sum + 4.5 * (vxz2 + vxz) - square)

                f[11] += (feq11 - f[11]) * tau_inv
                f[12] += (feq11 - 6.0 * dw[11] * rho * sum - f[12]) * tau_inv

                sum = vy + vz

                vyz = 2.0 * vy * vz

                vyz2 = vy * vy + vz * vz

                feq13 = dw[13] * rho * (1.0 + 3.0 * sum + 4.5 * (vyz + vyz2) - square)

                f[13] += (feq13 - f[13]) * tau_inv
                f[14] += (feq13 - 6.0 * dw[13] * rho * sum - f[14]) * tau_inv

                sum = vz - vx

                feq15 = dw[15] * rho * (1.0 + 3.0 * sum + 4.5 * (vxz2 - vxz) - square)

                f[15] += (feq15 - f[15]) * tau_inv
                f[16] += (feq15 - 6.0 * dw[15] * rho * sum - f[16]) * tau_inv

                sum = vz - vy

                feq17 = dw[17] * rho * (1.0 + 3.0 * sum + 4.5 * (vyz2 - vyz) - square)

                f[17] += (feq17 - f[17]) * tau_inv
                f[18] += (feq17 - 6.0 * dw[17] * rho * sum - f[18]) * tau_inv

                for i in range(19):
                    ftemp[x, y, i] = round(f[i] * 30000 / dw[i])
            else:
                for i in range(19):
                    ftemp[x, y, i] = 0

    ftable = sa.attach(ft1)
    ftable_empty = sa.attach(ft2)
    state = sa.attach(state)
    stream1 = cuda.stream()
    stream2 = cuda.stream()

    i1 = 205
    i2 = 595
    i3 = 799
    i4 = 800

    st = end - 1

    dabove = cuda.to_device(ftable[1, :500], stream=stream1)
    dcur = cuda.to_device(ftable[0, :500], stream=stream1)

    dtemp = cuda.device_array_like(dcur, stream=stream1)

    propagate_kernel[blockspergrid, threadsperblock, stream1](dtemp, dcur, dabove, dcur, tau_inv, out_u_lb, 1)
	
    while 1:
	
        dnext = cuda.to_device(ftable[2, :500], stream=stream2)

        stream2.synchronize()
        stream1.synchronize()
        dfirst = dtemp
        dtemp.copy_to_host(ftable_empty[0, :500], stream=stream2)

        dbelow = dcur
        dcur = dabove
        dabove = dnext

        for i in range(1, end):

            dtemp = cuda.device_array_like(dcur, stream=stream1)

            if i > i4:
                propagate_kernel_new[blockspergrid, threadsperblock, stream1](dtemp, dcur, dabove, dbelow, tau_inv)
            elif i1 < i < i2:
                propagate_kernel[blockspergrid, threadsperblock, stream1](dtemp, dcur, dabove, dbelow, tau_inv, state[1], 2)
            else:
                propagate_kernel[blockspergrid, threadsperblock, stream1](dtemp, dcur, dabove, dbelow, tau_inv, 0, 0)

            if i1 < i+2 < i2:
                dnext = cuda.to_device(ftable[i + 2, :600], stream=stream2)
            elif i+2 < i3:
                dnext = cuda.to_device(ftable[i + 2, :500], stream=stream2)
            elif i < st:
                dnext = cuda.to_device(ftable[i + 2], stream=stream2)

            stream2.synchronize()
            stream1.synchronize()
            dtemp.copy_to_host(ftable_empty[i, :dtemp.shape[0]], stream=stream2)

            dbelow = dcur
            dcur = dabove
            dabove = dnext

        stream2.synchronize()

        temp = ftable
        ftable = ftable_empty
        ftable_empty = temp

        dabove = cuda.to_device(ftable[1, :500], stream=stream1)
        dcur = dfirst
    
        dtemp = cuda.device_array_like(dcur, stream=stream1)
    
        propagate_kernel[blockspergrid, threadsperblock, stream1](dtemp, dcur, dabove, dcur, tau_inv, out_u_lb, 1)

        cond.acquire()
        state[0] += 1

        if state[0] == device_num:
            event.set()

        cond.wait()
        cond.release()



# thread function to handle streaming step
def propagate(ft1, ft2, height, blockspergrid, threadsperblock, gpu_id, start, end, tau_inv, state, event, cond, device_num):

    cuda.select_device(gpu_id)

    # thread function to handle streaming step
    @cuda.jit('void(uint16[:,:,:], uint16[:,:,:], uint16[:,:,:], uint16[:,:,:], float32)')
    def propagate_kernel_new(ftemp, dcur, dabove, dbelow, tau_inv):

        # set constant array of direction vectors
        de = cuda.const.array_like(e)
        # set constant array of opposite direction indices
        dopp = cuda.const.array_like(opp)

        dopp2 = cuda.const.array_like(opp2)
        # obtain position in grid
        dw = cuda.const.array_like(w)

        x, y = cuda.grid(2)

        # if within domain and  not solid node
        if x < dcur.shape[0] and y < dcur.shape[1]:

            ftemp[x, y, 0] = dcur[x, y, 0]

            for i in range(1, 19):
                # obtain original position of f distribution
                xi = x + de[i, 1]
                yi = y - de[i, 2]

                if yi == -1:
                    yi = dcur.shape[1] - 1
                elif yi == dcur.shape[1]:
                    yi = 0

                if x == dcur.shape[0] - 1:

                    # if from current layer
                    if de[i, 0] == 0:

                        if xi == dcur.shape[0]:
                            j = dopp[i]
                            ftemp[x, y, i] = dcur[x, y, j]
                        else:
                            ftemp[x, y, i] = dcur[xi, yi, i]

                    # if from layer below
                    elif de[i, 0] > 0:

                        if xi == dcur.shape[0]:
                            j = dopp[i]
                            ftemp[x, y, i] = dcur[x, y, j]
                        else:
                            ftemp[x, y, i] = dbelow[xi, yi, i]
                    # if from layer above
                    elif de[i, 0] < 0:

                        if xi == dcur.shape[0]:
                            j = dopp[i]
                            ftemp[x, y, i] = dcur[x, y, j]
                        else:
                            ftemp[x, y, i] = dabove[xi, yi, i]

                elif x == 0:

                    # if from current layer
                    if de[i, 0] == 0:

                        if xi == -1:
                            j = dopp2[i]
                            ftemp[x, y, i] = dcur[x, yi, j]
                        else:
                            ftemp[x, y, i] = dcur[xi, yi, i]

                    # if from layer below
                    elif de[i, 0] > 0:

                        if xi == -1:
                            j = dopp2[i]
                            ftemp[x, y, i] = dbelow[x, yi, j]
                        else:
                            ftemp[x, y, i] = dbelow[xi, yi, i]
                    # if from layer above
                    elif de[i, 0] < 0:

                        if xi == -1:
                            j = dopp2[i]
                            ftemp[x, y, i] = dabove[x, yi, j]
                        else:
                            ftemp[x, y, i] = dabove[xi, yi, i]

                else:

                    # if from current layer
                    if de[i, 0] == 0:
                        ftemp[x, y, i] = dcur[xi, yi, i]
                    # if from layer below
                    elif de[i, 0] > 0:
                        ftemp[x, y, i] = dbelow[xi, yi, i]
                    # if from layer above
                    elif de[i, 0] < 0:
                        ftemp[x, y, i] = dabove[xi, yi, i]

            f = cuda.local.array(19, dtype=float32)

            rho = 0.0

            for i in range(19):
                f[i] = ftemp[x, y, i] / 30000 * dw[i]
                rho += f[i]

            vx = (f[1] - f[2] + f[7] - f[8] + f[10] - f[9] + f[11] - f[12] + f[16] - f[15]) / rho

            vy = (f[3] - f[4] + f[7] - f[8] + f[9] - f[10] + f[13] - f[14] + f[18] - f[17]) / rho

            vz = (f[5] - f[6] + f[11] - f[12] + f[13] - f[14] + f[15] - f[16] + f[17] - f[18]) / rho

            # calculate feq and f after collision

            square = 1.5 * (vx * vx + vy * vy + vz * vz)

            f[0] += (dw[0] * rho * (1 - square) - f[0]) * tau_inv

            feq1 = dw[1] * rho * (1.0 + 3.0 * vx + 4.5 * vx * vx - square)

            f[1] += (feq1 - f[1]) * tau_inv

            f[2] += (feq1 - 6.0 * dw[1] * rho * vx - f[2]) * tau_inv

            feq3 = dw[3] * rho * (1.0 + 3.0 * vy + 4.5 * vy * vy - square)

            f[3] += (feq3 - f[3]) * tau_inv

            f[4] += (feq3 - 6.0 * dw[3] * rho * vy - f[4]) * tau_inv

            feq5 = dw[5] * rho * (1.0 + 3.0 * vz + 4.5 * vz * vz - square)

            f[5] += (feq5 - f[5]) * tau_inv

            f[6] += (feq5 - 6.0 * dw[5] * rho * vz - f[6]) * tau_inv

            sum = vx + vy

            vxy = 2.0 * vx * vy

            vxy2 = vx * vx + vy * vy

            feq7 = dw[7] * rho * (1.0 + 3.0 * sum + 4.5 * (vxy2 + vxy) - square)

            f[7] += (feq7 - f[7]) * tau_inv
            f[8] += (feq7 - 6.0 * dw[7] * rho * sum - f[8]) * tau_inv

            sum = vy - vx

            feq9 = dw[9] * rho * (1.0 + 3.0 * sum + 4.5 * (vxy2 - vxy) - square)

            f[9] += (feq9 - f[9]) * tau_inv
            f[10] += (feq9 - 6.0 * dw[9] * rho * sum - f[10]) * tau_inv

            sum = vx + vz

            vxz = 2.0 * vx * vz

            vxz2 = vx * vx + vz * vz

            feq11 = dw[11] * rho * (1.0 + 3.0 * sum + 4.5 * (vxz2 + vxz) - square)

            f[11] += (feq11 - f[11]) * tau_inv
            f[12] += (feq11 - 6.0 * dw[11] * rho * sum - f[12]) * tau_inv

            sum = vy + vz

            vyz = 2.0 * vy * vz

            vyz2 = vy * vy + vz * vz

            feq13 = dw[13] * rho * (1.0 + 3.0 * sum + 4.5 * (vyz + vyz2) - square)

            f[13] += (feq13 - f[13]) * tau_inv
            f[14] += (feq13 - 6.0 * dw[13] * rho * sum - f[14]) * tau_inv

            sum = vz - vx

            feq15 = dw[15] * rho * (1.0 + 3.0 * sum + 4.5 * (vxz2 - vxz) - square)

            f[15] += (feq15 - f[15]) * tau_inv
            f[16] += (feq15 - 6.0 * dw[15] * rho * sum - f[16]) * tau_inv

            sum = vz - vy

            feq17 = dw[17] * rho * (1.0 + 3.0 * sum + 4.5 * (vyz2 - vyz) - square)

            f[17] += (feq17 - f[17]) * tau_inv
            f[18] += (feq17 - 6.0 * dw[17] * rho * sum - f[18]) * tau_inv

            for i in range(19):
                ftemp[x, y, i] = round(f[i] * 30000 / dw[i])

    ftable = sa.attach(ft1)
    ftable_empty = sa.attach(ft2)
    state = sa.attach(state)
    stream1 = cuda.stream()
    stream2 = cuda.stream()

    st = end - 1 if end < height else height - 2

    dabove = cuda.to_device(ftable[start + 1], stream=stream1)
    dcur = cuda.to_device(ftable[start], stream=stream1)
    dtemp = cuda.device_array_like(ftable_empty[0], stream=stream1) 

    while 1:

        dbelow = cuda.to_device(ftable[start - 1], stream=stream1)
        dfirst = dtemp
        dsecond = dbelow

        for i in range(start, end):

            propagate_kernel_new[blockspergrid, threadsperblock, stream1](dtemp, dcur, dabove, dbelow, tau_inv)

            if i < st:
                dnext = cuda.to_device(ftable[i + 2], stream=stream2)

            stream2.synchronize()
            stream1.synchronize()
            dtemp.copy_to_host(ftable_empty[i], stream=stream2)

            dtemp = dbelow
            dbelow = dcur
            dcur = dabove
            dabove = dnext

        stream2.synchronize()

        temp = ftable
        ftable = ftable_empty
        ftable_empty = temp

        dabove = dsecond
        dcur = dfirst
    
        cond.acquire()
        state[0] += 1

        if state[0] == device_num:
            event.set()

        cond.wait()
        cond.release()



# thread function to handle streaming step
def normalization(ft1, blockspergrid, threadsperblock, divisor):

    cuda.select_device(0)

    # thread function to handle streaming step
    @cuda.jit('void(uint16[:,:,:], float32)')
    def round_down(ftemp, divisor):

        x, y = cuda.grid(2)

        # if within domain and  not solid node
        if x < ftemp.shape[0] and y < ftemp.shape[1]:

            for i in range(19):
                ftemp[x, y, i] = round(ftemp[x, y, i] / divisor)

    ftable = sa.attach(ft1)
    stream1 = cuda.stream()
    stream2 = cuda.stream()

    height = ftable.shape[0]
    dtemp = cuda.to_device(ftable[0], stream=stream2)
    stream2.synchronize()
    for i in range(height):

        round_down[blockspergrid, threadsperblock, stream1](dtemp, divisor)

        if i < height - 1:
            dnext = cuda.to_device(ftable[i + 1], stream=stream2)

        stream2.synchronize()
        stream1.synchronize()
        dtemp.copy_to_host(ftable[i], stream=stream2)

        dtemp = dnext

    stream2.synchronize()



if __name__ == '__main__':

    # dx between lattice nodes in m
    lu = 0.02 / 200

    # maximum physical diaphram velocity
    dia_u_p = 0.07
    # maximum physical side flow velocity
    out_u_p = 0.5
    # duration of simulation in s
    duration = 173 / 1000
    # diaphragm oscillation frequency
    frequency = 40

    # data output interval
    interval = 5 / 1000
    # max LB velocity
    max_u_lb = 0.2

    max_u_p = 1.5
    # physical k viscosity
    Nu_P = 15 / 1000000
    # nozzle model file name
    model_fileName = 'sja.STL'

    outer_flow_direction = 'left to right'

    dia_delay = 2.78

    device_num = 4

    file_name = 'result3ftable.npy'
    out_folder = 'result3'

    with open('result3time.dat', 'r') as f:
        t = float(f.readline())

    ts = lu * max_u_lb / abs(max_u_p)

    if abs(out_u_p) / (lu / ts) > 0.2:
        print('External Flow Speed Exceed 0.2 u')

    Nulb = lu * lu / ts

    dia_u_lb_max = dia_u_p * ts / lu

    if outer_flow_direction == 'no flow':
        out_u_lb = 0
    else:
        out_u_lb = out_u_p * ts / lu

    print("Time Step: {} s".format(ts))

    tau_inv = 1.0 / ((Nu_P / Nulb) * 3.0 + 0.5)

    print("tau_inverse: {}".format(tau_inv))

    ftable = np.load(file_name)

    for x in sa.list():
        sa.delete(x.name.decode("utf-8"))

    amx = np.amax(ftable)

    ftable1 = sa.create("shm://ftable1", ftable.shape, np.uint16)

    np.copyto(ftable1, ftable)

    ftable = ftable1

    ft1 = "ftable1"
    ft2 = "ftable2"

    ftable_empty = sa.create("shm://ftable2", ftable.shape, np.uint16)

    # define GPU setting   threads per block  and  blocks per grid   for 1 layer along y axis
    threadsperblock = (16, 16)
    blockspergrid_y = math.ceil(ftable.shape[1] / threadsperblock[0])
    # blockspergrid_y2 = math.ceil(ftable.shape[1] / threadsperblock[0] / 2)
    blockspergrid_z = math.ceil(ftable.shape[2] / threadsperblock[1])
    blockspergrid = (blockspergrid_y, blockspergrid_z)
    # blockspergrid2 = (blockspergrid_y2, blockspergrid_z)
    print("ready")
    # initialize time and number of data output

    print(amx)

    if amx > 56000:
        p = multiprocessing.Process(target=normalization, args=(ft1, blockspergrid, threadsperblock, amx/53000))
        p.start()
        p.join()

    starttime = t

    output_num = starttime // interval + 1

    endtime = duration + starttime

    start = np.zeros(device_num, np.int32)
    end = np.zeros(device_num, np.int32)

    dia_delay -= ts

    for gpu_id in range(device_num):
        start[gpu_id] = gpu_id * (ftable.shape[0]) // device_num
        end[gpu_id] = (gpu_id + 1) * (ftable.shape[0]) // device_num

    event = multiprocessing.Event()
    cond = multiprocessing.Condition()

    state = sa.create("shm://state", 2, np.float32)

    state[0] = 0
    state[1] = 0 if t < dia_delay else dia_u_lb_max * np.sin(2 * np.pi * (t - dia_delay) * frequency)

    p_list = list()
    p_list.append(multiprocessing.Process(target=propagate_start, args=(ft1, ft2, blockspergrid,
                            threadsperblock, 0, end[0], out_u_lb, tau_inv, "state", event, cond, device_num)))

    for i in range(1, device_num):
        p_list.append(multiprocessing.Process(target=propagate, args=(ft1, ft2, ftable.shape[0], blockspergrid,
                                threadsperblock, i, start[i], end[i], tau_inv, "state", event, cond, device_num)))

    for p in p_list:
        p.start()

    while 1:

        temp = ftable

        ftable = ftable_empty

        ftable_empty = temp

        event.wait()

        t += ts

        state[1] = 0 if t < dia_delay else dia_u_lb_max * np.sin(2 * np.pi * (t - dia_delay) * frequency)

        # output data result after specific interval
        if t >= output_num * interval:

            output = multiprocessing.Process(target=out_result.output_result, args=(lu, ts,
                                                                                    ftable[:, :,
                                                                                    math.floor(ftable.shape[2] / 2), :],
                                                                                    out_folder, t))
            output.start()

            output = multiprocessing.Process(target=out_result.output_result_2, args=(lu, ts,
                                                                                      ftable[:, 750, :, :],
                                                                                      out_folder + 'h=750', t))

            output.start()

            output = multiprocessing.Process(target=out_result.output_result_2, args=(lu, ts,
                                                                                      ftable[:, 470, :, :],
                                                                                      out_folder + 'h=470', t))

            output.start()

            output = multiprocessing.Process(target=out_result.output_result_2, args=(lu, ts,
                                                                                      ftable[800:, 999, :, :],
                                                                                      out_folder + 'h=999', t))
            output.start()


            output_num += 1

        cond.acquire()

        if t > endtime:
            break

        state[0] = 0

        cond.notify_all()

        cond.release()

        event.clear()

    for p in p_list:
        p.terminate()

    np.save(out_folder + 'ftable', ftable)

    with open(out_folder + 'time.dat', 'w+') as f:
        f.write('{}\n'.format(t))

    for x in sa.list():
        sa.delete(x.name.decode("utf-8"))


