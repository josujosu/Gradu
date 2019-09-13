import struct
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as C
from ketjusnap_plotter import *
sys.path.append("../ketjugw/ketjugw")
import ketjugw

# ------ Unit conversion stuff --------

Msun_in_kg = 1.9885e30

# internal unit system
unit_mass_in_kg = Msun_in_kg
unit_vel_in_m_per_s = C.c

unit_length_in_m = C.G*Msun_in_kg/(C.c)**2
unit_length_in_au = unit_length_in_m / C.au
unit_length_in_pc = unit_length_in_m / C.parsec

unit_time_in_s = unit_length_in_m / unit_vel_in_m_per_s
unit_time_in_years = unit_time_in_s / C.year

# -------------------------------------

input_file = "ic_test_2.dat"
input_template = "ic_{}.dat"

basic_bins = np.logspace(0.5, 5.5, 100)

n_star = 415000
n_dm = 1000000

# --------- File reading stuff -------------

def read_file(input_file, n_dm, n_star):

    with open(input_file, "rb") as f:
        byte = f.read(4 * 3 * n_dm)

        arr_dm = [struct.unpack(str(3 * n_dm)+'f', byte)]

        byte = f.read(4 * 3 * n_star)
        arr_star = [struct.unpack(str(3 * n_star) + 'f', byte)]

        arr = np.append(arr_dm, arr_star)

    return arr


def get_pos_from_file(input_file, n_dm, n_star):

    arr = read_file(input_file, n_dm, n_star)

    return arr[:n_dm*3], arr[n_dm*3:3*(n_dm+n_star)]

def combine_multiple_pos(N, n_dm, n_star):

    pos = np.array([])

    for i in range(N):
        dummy, pos_dot = get_pos_from_file(input_template.format(i+1), n_dm, n_star)
        pos = np.append(pos, pos_dot)

    return pos

# -------------------------------------------------------------

def create_N_particle_bins(r, p_in_bin):
    return r[::p_in_bin]



# ----------------------------------------------------------
def save_2D_density_profile(input_file, n_dm, n_star, bins):

    dm_pos, star_pos = get_pos_from_file(input_file, n_dm, n_star)
    coords = star_pos.reshape((-1, 3))

    masses = np.ones(len(coords[:])) * 4.15e11 / n_star

    dens, r = mean_surface_density_profile(coords, masses, bins, 100)

    np.savetxt("ic_2D_profile.txt", np.column_stack((r/1000, dens)))


def save_density_profile(input_file, n_dm, n_star, bins, combination=0):

    if combination == 0:
        dm_pos, star_pos = get_pos_from_file(input_file, n_dm, n_star)
    else:
        star_pos = combine_multiple_pos(combination, n_dm, n_star)
        n_star = n_star * combination

    coords = star_pos.reshape((-1, 3))
    masses = np.ones(len(coords[:])) * 4.15e11 / n_star
    com = find_com(coords, masses, max(coords[:, 0]) * 0.001)
    coords = coords - com

    r = np.sqrt(np.sum(coords**2, axis=1))
    r = np.sort(r) * ketjugw.units.DataUnits().unit_length * unit_length_in_pc

    hist = np.histogram(r, bins=bins, weights=masses)

    densities = hist[0] / (4/3*np.pi * hist[1][1:]**3 - 4/3*np.pi * hist[1][:-1]**3)
    r = (hist[1][:-1] + hist[1][1:])/2

    I = densities
    r = r/1000

    np.savetxt("ic_profile.txt", np.column_stack((r, I)))


def plot_from_file(input_file):

    matplotlib.rcParams.update({"font.size": 18})

    data = np.genfromtxt(input_file)
    r = data[:,0]
    I = data[:,1]

    plt.plot(r, I, linewidth=3)

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-2, 1e2)
    plt.ylim(1e-6, 1e5)

    plt.ylabel("$\\rho(r) \; [M_\odot \mathrm{pc^{-3}}]$")
    plt.xlabel("$r \mathrm{[kpc]}$")

    plt.tick_params(which='both', top=True, right=True, left=True, direction='in', length=4)
    plt.tick_params(which='major', length=8)

    plt.show()


save_density_profile(input_file, n_dm, n_star, basic_bins)
plot_from_file("ic_profile.txt")


