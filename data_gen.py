import numpy as np
import pandas as pd
import random
import os
import argparse
import warnings
from multiprocessing import Pool, Manager

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Data generator')
parser.add_argument('--num_workers', type=int,
                    default=60, help='num workers')
parser.add_argument('--num_data', type=int,
                    default=300000, help='num data')
args = parser.parse_args()

Deg2Rad = np.pi/180
Rad2Deg = 1/Deg2Rad
g = 9.8
K_alt = .8*2          # hdot loop gain
RoC = 20              # maximum rate of climb (max. of hdot)
AoA0 = -1.71*Deg2Rad     # zero lift angle of attack
Acc2AoA = 0.308333*Deg2Rad  # 1m/s^2 ACC corresponds to 0.308333deg AOA
zeta_ap = 0.7         # pitch acceleration loop damping
omega_ap = 4          # pitch acceleration loop bandwidth
Vm_start = 1        # beginning point of vm range
Vm_end = 5         # end point of vm range
Vt_start = 1        # beginning point of vt range
Vt_end = 5         # end point of vmtrange

def work(number_of_loops, result_queue):
    work_result = uni_data_generator(number_of_loops)
    result_queue.put(work_result)
    return

def multiprocess_data_gen(num_workers_in, number_of_data_in):
    number_of_works = int(number_of_data_in/num_workers_in)
    pool = Pool(num_workers_in)
    m = Manager()
    result_queue = m.Queue()
    pool.starmap(work, [(number_of_works, result_queue) for _ in range(num_workers_in)])
    pool.close()
    pool.join()
    return result_queue

# hdot loop dynamics definition

def model(z, t, hdot_cmd, Vm):                          # computes state derivatives
    # state vector: a (pitch acc), adot, h (alt), hdot, R (ground-track range)
    a, adot, h, hdot, R = z
    gamma = np.arcsin(hdot/Vm)                          # fight path angle
    # pitch acceleration command
    ac = K_alt * (hdot_cmd - hdot) + g/np.cos(gamma)
    ac = np.clip(ac, -30, 30)                         # maneuver limit

    addot = omega_ap*omega_ap*(ac-a) - 2*zeta_ap*omega_ap*adot
    hddot = a*np.cos(gamma) - g
    Rdot = Vm*np.cos(gamma)
    # returns state derivatives
    return np.array([adot, addot, hdot, hddot, Rdot])


def data_gen():

    dt = 0.1              # control frequency
    tf = 40               # final time
    t = np.arange(0, tf, dt)
    stop_point = random.randrange(6, len(t))
    N = len(t)
    loop_continue = 1
    while loop_continue == 1:
        hm0 = 1000                                                         # initial altitude
        Vm = np.random.randint(Vm_start, Vm_end)                           # initial speed
        Vt = np.random.randint(Vt_start, Vt_end)                           # initial speed
        dist_sep = (Vm + Vt)*2                                             # near mid-air collision range (avoidance finished about 10 secs)
        # initial flight path angle                                                                       (based on assumption pitch angle is up to 30 degree)
        gamma0 = 0*Deg2Rad
        # initial NED position
        Pm_NED = np.array([30*Vt - 30*Vm, 0, -hm0])
        # initial NED velocity
        Vm_NED = np.array([Vm*np.cos(gamma0), 0, -Vm*np.sin(gamma0)])

        # state variable: [a, adot, h, hdot, R]
        X0 = np.array([g/np.cos(gamma0), 0, hm0, -Vm_NED[2], 30*Vt - 30*Vm]
                      )       # initial state vector

        # target initial conditions
        ht0 = 1000 + dist_sep*0.7*np.random.randn()
        approach_angle = 180*Deg2Rad*(2*np.random.rand()-1)
        psi0 = np.pi + approach_angle + 2*np.random.randn()*Deg2Rad
        psi0 = np.arctan2(np.sin(psi0), np.cos(psi0))

        Pt_N = 30*Vt*(1+np.cos(approach_angle))
        Pt_E = 30*Vt*np.sin(approach_angle)
        Pt_D = -ht0
        # initial NED position
        Pt_NED = np.array([Pt_N, Pt_E, Pt_D])
        # initial NED velocity
        Vt_NED = np.array([Vt*np.cos(psi0), Vt*np.sin(psi0), 0])

        # initialize variables
        X = np.zeros((N, len(X0)))
        X[0, :] = X0
        dotX_p = 0

        theta0 = gamma0 + X0[0]*Acc2AoA + AoA0  # initial pitch angle

        DCM = np.zeros((3, 3))                      # initial DCM NED-to-Body
        DCM[0, 0] = np.cos(theta0)
        DCM[0, 2] = -np.sin(theta0)
        DCM[1, 1] = 1
        DCM[2, 0] = np.sin(theta0)
        DCM[2, 2] = np.cos(theta0)

        Pr_NED = Pt_NED - Pm_NED                   # relative NED position
        Vr_NED = Vt_NED - Vm_NED                   # relative NED velosity

        # relative position (Body frame)
        Pr_Body = np.dot(DCM, Pr_NED)

        # radar outputs
        r = np.linalg.norm(Pr_Body)                # range
        vc = -np.dot(Pr_NED, Vr_NED)/r             # closing velocity
        # target vertival look angle (down +)
        elev = np.arctan2(Pr_Body[2], Pr_Body[0])
        # target horizontal look angle (right +)
        azim = np.arctan2(Pr_Body[1], Pr_Body[0]/np.cos(theta0))

        los = theta0 - elev                        # line of sight angle
        dlos = 0
        daz = 0

        # static variables
        los_p = los
        dlos_p = dlos
        azim_p = azim
        daz_p = daz
        hdot_cmd = 0
        hdot = 0
        gamma = gamma0
        count_change_hdot = 0

        # main loop
        for k in range(N-1):
            ##############################################################################
            # UPDATE ENVIRONMENT AND GET OBSERVATION

            # update environment
            # adams-bashforth 2nd order integration
            dotX = model(X[k, :], t[k], hdot_cmd, Vm)
            X[k+1, :] = X[k, :] + 0.5*(3*dotX-dotX_p)*dt
            dotX_p = dotX

            Pt_NED = Pt_NED + Vt_NED*dt        # target position integration

            # get observation

            a, adot, h, hdot, R = X[k+1, :]

            gamma = np.arcsin(hdot/Vm)
            theta = gamma + a*Acc2AoA + AoA0

            DCM = np.zeros((3, 3))
            DCM[0, 0] = np.cos(theta)
            DCM[0, 2] = -np.sin(theta)
            DCM[1, 1] = 1
            DCM[2, 0] = np.sin(theta)
            DCM[2, 2] = np.cos(theta)

            Pm_NED = np.array([R, 0, -h])
            Vm_NED = np.array([Vm*np.cos(gamma), 0, -Vm*np.sin(gamma)])

            Pr_NED = Pt_NED - Pm_NED
            Vr_NED = Vt_NED - Vm_NED

            Pr_Body = np.dot(DCM, Pr_NED)

            r = np.linalg.norm(Pr_Body)
            vc = -np.dot(Pr_NED, Vr_NED)/r
            elev = np.arctan2(Pr_Body[2], Pr_Body[0])
            azim = np.arctan2(Pr_Body[1], Pr_Body[0]/np.cos(theta))

            psi = np.arctan2(Vt_NED[1], Vt_NED[0])

            # los rate and az rate estimation
            los = theta - elev

            # filtered LOS rate, F(s)=20s/(s+20)
            dlos = (30*(los-los_p) + 0*dlos_p) / 3
            # filtered azim rate, F(s)=20s/(s+20)
            daz = (30*(azim-azim_p) + 0*daz_p) / 3

            los_p = los
            dlos_p = dlos
            azim_p = azim
            daz_p = daz

            # estimate closest approach
            min_dist_vert = r*r/vc*dlos
            min_dist_horiz = r*r/vc*daz

            # estimate cruise distance
            dist_cruise = r*los
            t_col = -np.dot(Vr_NED,Pr_NED)/np.linalg.norm(Vr_NED)/np.linalg.norm(Vr_NED)
            closest_NED = Pr_NED + Vr_NED*t_col
            zem = np.linalg.norm(closest_NED)
            zem_v2 = -closest_NED[-1]     
            zem_h2 = zem                 

            Vm_NED_c = np.array([Vm_NED[0], Vm_NED[1], 0])
            Vm_NED_c = Vm_NED_c*np.linalg.norm(Vm_NED)/np.linalg.norm(Vm_NED_c)
            Vr_NED_c = Vt_NED - Vm_NED_c
            t_col_c = -np.dot(Vr_NED_c,Pr_NED)/np.linalg.norm(Vm_NED_c)/np.linalg.norm(Vm_NED_c)
            closest_NED_c = Pr_NED + Vr_NED_c*t_col_c
            crm_c = np.linalg.norm(closest_NED_c)
            crm_v2 = -closest_NED_c[-1]   
            ##############################################################################
            # COMPUTE ACTION (BEGIN)
            if k>5 and r > dist_sep:
                if zem_v2 > 0:
                    if zem_v2 < dist_sep:
                        if np.abs(crm_v2) < dist_sep:
                            if (zem_h2 < dist_sep):
                                if hdot_cmd != -int(dist_sep/15):
                                    count_change_hdot += 1

                                hdot_cmd = -int(dist_sep/15)
                            else:
                                hdot_cmd = 0
                        else:
                            hdot_cmd = 0
                    else:
                        if np.abs(crm_v2) > dist_sep:
                            hdot_cmd = 0
                        else:
                            if hdot_cmd != -int(dist_sep/15):
                                count_change_hdot += 1
                            hdot_cmd = -int(dist_sep/15)
                else:
                    if zem_v2 > -dist_sep:
                        if np.abs(crm_v2) < dist_sep:
                            if zem_h2 < dist_sep:
                                if hdot_cmd != int(dist_sep/15):
                                    count_change_hdot += 1

                                hdot_cmd = int(dist_sep/15)
                            else:
                                hdot_cmd = 0
                        else:
                            hdot_cmd = 0
                    else:
                        if np.abs(crm_v2) > dist_sep:
                            hdot_cmd = 0
                        else:
                            if hdot_cmd != int(dist_sep/15):
                                count_change_hdot += 1
                            hdot_cmd = int(dist_sep/15)
                if k == stop_point:
                    # WRITE DATA
                    loop_continue = 0
                    break
            if vc < 0 or t_col > 20:
                hdot_cmd = 0
                loop_continue = 0
                break

    if hdot_cmd == -20:
        hdot_cmd = 1
    if hdot_cmd == 0:
        hdot_cmd = 0
    if hdot_cmd == 20:
        hdot_cmd = 2
    return [r, vc, azim, los, daz, dlos, hdot_cmd]


def uni_data_generator(num_of_data):
    data = []
    each_num = num_of_data/3
    all_feat = 0
    down = 0
    stay = 0
    up = 0
    while(all_feat < 3):
        temp = data_gen()
        if (temp[-1] == 0 and down < each_num):
            data.append(temp)
            down += 1
        elif (temp[-1] == 1 and stay < each_num):
            data.append(temp)
            stay += 1
        elif (temp[-1] == 2 and up < each_num):
            data.append(temp)
            up += 1
        all_feat = (down+stay+up)/each_num
    return data


if __name__ == "__main__":
    num_workers = args.num_workers
    number_of_data = args.num_data
    if os.path.exists('norm_data_train_uniform_ext.csv') is False:
        # train data
        data = multiprocess_data_gen(num_workers, number_of_data)
        data.put("stop")
        res = []
        while True:
            data_piece = data.get()
            if data_piece == "stop":
                break
            else:
                res = res + data_piece
        res = np.array(res)
        mean = res.mean(axis=0)[:6]
        std = res.std(axis=0)[:6]
        np.save('mean.npy', [_ for _ in mean[:]])
        np.save('std.npy', [_ for _ in std[:]])
        res[:, :6] = (res[:, :6]-mean)/(std)
        df = pd.DataFrame(res)
        df.to_csv('norm_data_train_uniform_ext.csv', header=False, index=False)
        print("Train data generation complete")
    if os.path.exists('norm_data_test_uniform_ext.csv') is False:
        # validation data
        data = multiprocess_data_gen(num_workers, int(number_of_data/10))
        data.put("stop")
        res = []
        while True:
            data_piece = data.get()
            if data_piece == "stop":
                break
            else:
                res = res + data_piece
        res = np.array(res)
        mean = res.mean(axis=0)[:6]
        std = res.std(axis=0)[:6]
        np.save('mean_test.npy', [_ for _ in mean[:]])
        np.save('std_test.npy', [_ for _ in std[:]])
        res[:, :6] = (res[:, :6]-mean)/(std)
        df = pd.DataFrame(res)
        df.to_csv('norm_data_test_uniform_ext.csv', header=False, index=False)
        print("validation data generation complete")
