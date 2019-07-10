import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import scipy.constants as C
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.interpolation import spline_filter1d
sys.path.append("../ketjugw/ketjugw")
import ketjugw


# Unit conversions from units.py
Msun_in_kg = 1.9885e30

# internal unit system
unit_mass_in_kg = Msun_in_kg
unit_vel_in_m_per_s = C.c

unit_length_in_m = C.G*Msun_in_kg/(C.c)**2
unit_length_in_au = unit_length_in_m / C.au
unit_length_in_pc = unit_length_in_m / C.parsec

unit_time_in_s = unit_length_in_m / unit_vel_in_m_per_s
unit_time_in_years = unit_time_in_s / C.year


def smooth(xs, m):
    print("smoothing")
    res = np.zeros(len(xs)-2*m)
    for i in range(m, len(res)+m):
        res[i-m] = np.mean(xs[i-m:i+1+m])
    return res


def thin_data(data, c):
    mask = np.arange(0, len(data)) % c == 0
    return data[mask]


def vector_length(vec):
    return np.sqrt(vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2)

def com_coordinates(bh1, bh2):
    total_m = bh1.m + bh2.m
    r_cm = (bh1.x * bh1.m[0] + bh2.x * bh2.m[0]) / total_m[0]

    return -r_cm + bh1.x, -r_cm + bh2.x


def com_velocities(bh1, bh2):
    total_m = bh1.m + bh2.m
    v_cm = (bh1.v * bh1.m[0] + bh2.v * bh2.m[0]) / total_m[0]

    return -v_cm + bh1.v, -v_cm + bh2.v


def keplerian_semi_major_axis(bh1, bh2):
    rel_r = - bh1.x + bh2.x
    rel_v = - bh1.v + bh2.v
    return (2/vector_length(rel_r) - vector_length(rel_v)**2 / (bh1.m[0] + bh2.m[0]))**-1


def keplerian_eccentricity(bh1, bh2):
    rel_r = - bh1.x + bh2.x
    rel_v = - bh1.v + bh2.v
    r = np.sqrt(rel_r[:,0]**2 + rel_r[:,1]**2 + rel_r[:,2]**2)
    a = keplerian_semi_major_axis(bh1, bh2)
    dotp = rel_r[:,0] * rel_v[:,0] + rel_r[:,1] * rel_v[:,1] + rel_r[:,2] * rel_v[:,2]
    return np.sqrt((1-r/a)**2 + 1/a * (dotp**2)/(bh1.m[0] + bh2.m[0]))

# Evolution of the semi-major axis
def dadt(a, e, m1, m2):
    return -64/5 * m1*m2*(m1+m2) / (a**3*(1-e**2)**(7/2)) * (1 + 73/24 * e**2 + 37/96 * e**4)

# Evolution of the eccentricity (e_t)
def dedt(a, e, m1, m2):
    return -304/15 * e * m1*m2*(m1+m2) / (a**4*(1-e**2)**(5/2)) * (1 + 121/304 * e**2)


# -------------------------------------------------------

def plot_as_and_es(data_path, data_files):
    matplotlib.rcParams.update({'font.size': 16})

    f, (a0, a1) = plt.subplots(1, 2)

    for i, data_file in enumerate(data_files):
        bh1, bh2 = ketjugw.data_input.load_data(data_path + data_file,
                                                ketjugw.units.DataUnits()
                                                )

        elem = ketjugw.orbital_parameters(bh1, bh2)

        # e = orb_elements['e_t']
        da = dadt(elem['a_R'], elem['e_t'], bh1.m[0], bh2.m[0]) * unit_length_in_pc / unit_time_in_years
        t = (bh1.t - bh1.t[-1]) * unit_time_in_years / 1e9
        #t = 1 - (-bh1.t + bh1.t[-1]) / (-bh1.t[0] + bh1.t[-1])

        # plt.ylabel("Eccentricity")
        # plt.ylabel("|da/dt| [pc/yr]")
        # plt.xlabel("Time [Gyr]")
        # plt.plot(t, elem['a_R'] * unit_length_in_pc, label="Run {}".format(i + 1))

        a0.plot(t, elem['a_R'] * unit_length_in_pc, label="Run {}".format(i + 1))

        a1.plot(t, elem['e_t'], label="Run {}".format(i + 1))

    a0.set_xlabel("$T$ [yr]")
    a0.set_ylabel("$a$ [pc]")
    a0.semilogy()
    a0.legend(loc=3, fontsize=20)

    a1.set_xlabel("$T$ [yr]")
    a1.set_ylabel("$e$")
    a1.legend(loc=3, fontsize=20)

    plt.show()


def plot_trajectory(data_path, data_file):

    matplotlib.rcParams.update({'font.size': 14})

    bh1, bh2 = ketjugw.data_input.load_data(data_path + data_file,
                                                ketjugw.units.DataUnits()
                                                )

    com = (bh1.x * bh1.m[0] + bh2.x * bh2.m[0]) / (bh1.m[0] + bh2.m[0])

    x1 = (bh1.x - com[0,:]) * unit_length_in_pc
    x2 = (bh2.x - com[0,:]) * unit_length_in_pc

    plt.plot(x2[:,0], x2[:,1], label='$M_\\bullet = {:.2e} M_\odot$'.format(bh2.m[0] * unit_mass_in_kg / 1.989e30))
    plt.plot(x1[:,0], x1[:,1], label='$M_\\bullet = {:.2e} M_\odot$'.format(bh1.m[0] * unit_mass_in_kg / 1.989e30))
    plt.plot(0, 0, 'ro', label='Initial c.o.m. position')

    plt.legend(loc=3)
    plt.xlabel("x-position [pc]")
    plt.ylabel("y-position [pc]")
    plt.show()

data_path = "../data/data_dumps/"
data_files = [
    "chain_1.txt",
    "chain_2.txt",
    "chain_3.txt",
    "chain_4.txt",
#    "chain_new.txt"
    ]

plot_as_and_es(data_path, data_files)
#plot_trajectory(data_path, data_files[2])

'''
matplotlib.rcParams.update({'font.size': 14})

bh1, bh2 = ketjugw.data_input.load_data(data_path + data_files[3],
                                        ketjugw.units.DataUnits()
                                        )

xcom1, xcom2 = com_coordinates(bh1, bh2)

print(len(xcom2))

xcom2 = xcom2[-int(len(xcom2)/10000):]
xcom1 = xcom1[-int(len(xcom1)/10000):]


x1 = thin_data(xcom1[:,0], 1) * unit_length_in_pc
x2 = thin_data(xcom2[:,0], 1) * unit_length_in_pc

y1 = thin_data(xcom1[:,1], 1) * unit_length_in_pc
y2 = thin_data(xcom2[:,1], 1) * unit_length_in_pc

#plt.plot(x1, y1, 'ob', markersize=0.5)
#plt.plot(x2, y2, 'or', markersize=0.5)

x = bh2.x - bh1.x
x = x[-int(len(x)/1000):]
x = thin_data(x, 10)

plt.plot(x[:,0], x[:,1])

plt.xlabel("$x$ [pc]")
plt.ylabel("$y$ [pc]")
plt.show()
'''