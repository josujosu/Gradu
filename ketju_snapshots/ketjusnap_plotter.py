import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py as h5
import sys
import scipy.constants as C
from scipy.optimize import least_squares
from scipy.optimize import minimize
sys.path.append("../ketjugw/ketjugw")
import ketjugw

# -------------- Unit stuff ---------------

Msun_in_kg = 1.9885e30

# internal unit system
unit_mass_in_kg = Msun_in_kg
unit_vel_in_m_per_s = C.c

unit_length_in_m = C.G*Msun_in_kg/(C.c)**2
unit_length_in_au = unit_length_in_m / C.au
unit_length_in_pc = unit_length_in_m / C.parsec

unit_time_in_s = unit_length_in_m / unit_vel_in_m_per_s
unit_time_in_years = unit_time_in_s / C.year

def mu_in_mag_per_arcsec_squared(M_sun, I):
    return M_sun + 21.572 - 2.5 * np.log10(I)

def mu_in_lsun_per_pc_squared(M_sun, mu):
    return 10**((-mu+M_sun+21.572) / 2.5)


filenames = [
    "gamma-1.5-bh-0.hdf5",
    "gamma-1.5-bh-1.hdf5",
    "gamma-1.5-bh-2.hdf5",
    "gamma-1.5-bh-3.hdf5",
    "gamma-1.5-bh-4.hdf5",
    "gamma-1.5-bh-5.hdf5",
    "gamma-1.5-bh-6.hdf5"
]

basic_bins = np.logspace(np.log10(40), np.log10(62000), 50)
basic_bins20 = np.logspace(np.log10(40), np.log10(62000), 20)

nuker_bins = np.logspace(np.log10(40), np.log10(3100), 50)

beta_bins = np.logspace(1.9, 4.5, 20)
beta_bins_rb = np.logspace(1.4, 4.5, 20)
beta_bins2 = np.append(np.array([0, 100]), np.arange(190, 2e4, 0.19e3))

# Good
beta_bins3 = np.arange(50, 1.5*10**4.1, 0.10e3)
beta_bins3_rb = np.arange(0.1, 1.5*10**4.1, 0.10e3)

beta_bins_final = np.append(np.array([0, 100, 190]), np.logspace(np.log10(2.5*190), 5, 20))

# Line colors

colors = ['black', 'red', 'blue', 'orange', 'green', 'purple', 'slategray']


# ---------- Theoretical profile functions -----------


def b_n(n):
    return 2*n - 1/3


def sersic_profile(r, i_e, r_e, n=4):
    return i_e * np.exp(-b_n(n) * ((r/r_e)**(1/n)-1))


def core_sersic_profile(r, mu_b, r_b, alpha, r_e, gamma, n=4):

    #if gamma < 0: gamma = 0
    if gamma > 0.3:   gamma = 0.3

    mu_dot = mu_b * 2**(-gamma/alpha) * np.exp(b_n(n) * (2**(1/alpha) * r_b/r_e)**(1/n))

    return mu_dot * (1 + (r_b/r)**alpha)**(gamma/alpha) * \
           np.exp(-b_n(n) * (((r**alpha)+(r_b**alpha)) / (r_e**alpha))**(1/(alpha*n)))


def nuker_profile(r, mu_b, r_b, beta, alpha, gamma):

    #if gamma < 0: gamma = abs(gamma)

    return mu_b * 2**((beta-gamma)/alpha) * (r/r_b)**(-gamma) * (1 + (r/r_b)**alpha)**((gamma-beta)/alpha)

# ----------- Least-squares functions ---------------

# pars0 = i_e, pars1 = r_e, pars2 = m
def ls_sersic(pars, r, I):
    return (I - sersic_profile(r, pars[0], pars[1], pars[2])) / I


# pars0 = mu, pars1 = r_b, pars2 = alpha, pars3 = gamma, pars4 = r_e, pars5 = n
def ls_core_sersic(pars, r, I):
    return (I - core_sersic_profile(r, pars[0], pars[1], pars[2], pars[4], pars[3]))/I


# pars0 = mub, pars1 = rb, pars2 = beta, pars3 = alpha, pars4 = gamma
def ls_nuker(pars, r, I):
    return (I - nuker_profile(r, pars[0], pars[1], pars[2], pars[3], pars[4])) / I

# ------------ Effective radius finder ----------------


def find_effective_radius(I, r, bins):

    I_tot = np.sum(I * np.pi * (bins[1:]**2 - bins[:-1]**2))

    j = 0
    I_sum = I[j] * np.pi * (bins[j + 1] ** 2 - bins[j] ** 2)

    while I_sum < I_tot/2:
        j += 1
        I_sum += I[j] * np.pi * (bins[j+1]**2 - bins[j]**2)

    return r[j], I[j]

# ----------------------------------------------------
# ----------- Centre-of-mass methods -----------------

def find_com_of_sphere(coords, masses, r0, com0):
    com0_coords = coords - com0
    com0_dists = np.sqrt(com0_coords[:,0]**2 + com0_coords[:,1]**2 + com0_coords[:,2]**2)
    in_s = com0_dists < r0
    com_sphere_x = np.sum(coords[in_s,0] * masses[in_s]) / np.sum(masses[in_s])
    com_sphere_y = np.sum(coords[in_s,1] * masses[in_s]) / np.sum(masses[in_s])
    com_sphere_z = np.sum(coords[in_s,2] * masses[in_s]) / np.sum(masses[in_s])
    return np.array([com_sphere_x, com_sphere_y, com_sphere_z]), in_s


def find_com(coords, masses, r0):

    com, mask = find_com_of_sphere(coords, masses, r0, np.array([0, 0, 0]))
    new_coords = coords[mask]
    new_masses = masses[mask]
    r = r0 - r0*0.05

    while r > r0 * 0.05:

        com, mask = find_com_of_sphere(new_coords, new_masses, r, com)
        new_coords = new_coords[mask]
        new_masses = new_masses[mask]
        r -= r0 * 0.05

    return com


def com_velocity(masses, vels):
    numer = np.sum(vels * np.column_stack((masses, masses, masses)), axis=0)
    denumer = np.sum(masses)
    return numer / denumer

# --------------------------------------------------
# ---- Numerical profile calculation methods -------


def rotate_points(coords, alpha, beta, phi):
    x_rot = np.array([[1, 0, 0],
                      [0, np.cos(alpha), np.sin(alpha)],
                      [0, -np.sin(alpha), np.cos(alpha)]])
    y_rot = np.array([[np.cos(beta), 0, -np.sin(beta)],
                      [0, 1, 0],
                      [np.sin(beta), 0, np.cos(beta)]])
    z_rot = np.array([[np.cos(phi), np.sin(phi), 0],
                      [-np.sin(phi), np.cos(phi), 0],
                      [0, 0, 1]])
    return (x_rot @ y_rot @ z_rot) @ coords.T


# view angle:
#   viewing direction is alongside z-axis.
#       alpha: rotate points around x-axis
#       beta: rotate points around y-axis
#       phi: rotate points around z-axis
def surface_density_profile(coords, masses, bins, alpha=0, beta=0, phi=0):
    rot_coords = rotate_points(coords, alpha, beta, phi).T

    dists = np.sqrt(rot_coords[:,0]**2 + rot_coords[:,1]**2)

    mask = (dists < np.max(bins)) * (dists > np.min(bins))

    mass_hist = np.histogram(dists[mask],
                             bins=bins,
                             weights=masses[mask]
                             )

    densities = mass_hist[0] / (np.pi * mass_hist[1][1:]**2 - np.pi * mass_hist[1][:-1]**2)
    r = (mass_hist[1][:-1] + mass_hist[1][1:])/2
    return densities, r


def mean_surface_density_profile(coords, masses, bins, N):

    # Find c.o.m.
    com = find_com(coords, masses, max(coords[:,0])*0.001)
    coords = coords-com


    # First density profile
    angles = np.random.rand(3) * 2 * np.pi
    N_densities, r = surface_density_profile(coords, masses, bins, angles[0], angles[1], angles[2])

    if N == 1:
        return N_densities, r


    # Rest of the density profiles
    for i in range(N):
        print("- ", i)
        angles = np.random.rand(3) * 2 * np.pi
        densities, r = surface_density_profile(coords, masses, bins, angles[0], angles[1], angles[2])
        N_densities = np.column_stack((N_densities, densities))

    return np.mean(N_densities, axis=1), r


# ------------ Velocity disperison --------------

def cartesian_to_spherical(vectors):

    rho = np.sqrt(vectors[:,0]**2 + vectors[:,1]**2 + vectors[:,2]**2)
    theta = np.arctan2(np.sqrt(vectors[:,0]**2 + vectors[:,1]**2), vectors[:,2])
    phi = np.arctan2(vectors[:,1], vectors[:,0])

    return np.column_stack((rho, theta, phi))


# Rotates the vector in such a way that:
#   - x -> -theta
#   - y -> phi
#   - z -> r
def spherical_velocities(vels, coords):

    for i in range(len(vels[:])):
        print(i, "/", len(vels[:]))
        vels[i] = rotate_points(vels[i], 0, coords[i,1], coords[i,2])

    v_r = vels[:,2]
    v_th = vels[:,0]
    v_p = vels[:,1]

    return v_r, v_th, v_p


def project_vector_onto_plane(vectors, planes):
    p_lens = np.sqrt(planes[:, 0]**2 + planes[:, 1]**2 + planes[:, 2]**2)
    p_unit = planes / np.column_stack((p_lens, p_lens, p_lens))

    dot_p = np.sum(vectors * p_unit, axis=1)
    projected_par = np.column_stack((dot_p, dot_p, dot_p)) * p_unit
    projected_per = vectors - projected_par
    v_r = np.sqrt(projected_par[:, 0]**2 + projected_par[:, 1]**2 + projected_par[:, 2]**2)
    v_t = np.sqrt(projected_per[:, 0]**2 + projected_per[:, 1]**2 + projected_per[:, 2]**2)

    return v_r, v_t



def velocity_dispersion_profile(coords, masses, vels, bins, method='p'):

    # Find c.o.m.
    com = find_com(coords, masses, max(coords[:, 0]) * 0.001)
    coords = coords - com

    #v_com = find_com(vels, masses, max(vels[:,0]) * 0.001)
    #vels = vels - v_com

    dists = np.sqrt(coords[:,0]**2 + coords[:,1]**2 + coords[:,2]**2)

    # mask particles outside the bin-range
    mask = (dists < np.max(bins)) * (dists > np.min(bins))
    dists = dists[mask]
    coords = coords[mask]
    vels = vels[mask]

    # better bins?

    bins = np.sort(dists)[::int(dists.size/1000)]
    print(bins)

    r = (bins[1:] + bins[:-1]) / 2

    bin_mask = np.digitize(dists, bins)

    print(bin_mask)

    if method == 'rot':

        coords = cartesian_to_spherical(coords)

        vels = np.column_stack(spherical_velocities(vels, coords))

        sigma_r = np.zeros(len(r))
        sigma_th = np.zeros(len(r))
        sigma_p = np.zeros(len(r))

        for i in range(1, len(r) + 1):
            mask = bin_mask == i

            sigma_r[i - 1] = np.sqrt(np.mean(vels[mask, 0] ** 2))
            sigma_th[i - 1] = np.sqrt(np.mean(vels[mask, 1] ** 2))
            sigma_p[i - 1] = np.sqrt(np.mean(vels[mask, 2] ** 2))

        return r, np.column_stack((sigma_r, np.sqrt((sigma_th**2 + sigma_p**2)/2)))

    else:

        vels = np.column_stack(project_vector_onto_plane(vels, coords))

        sigma_r = np.zeros(len(r))
        sigma_t = np.zeros(len(r))

        for i in range(1, len(r) + 1):
            mask = bin_mask == i

            sigma_r[i - 1] = np.sqrt(np.mean(vels[mask, 0] ** 2))
            sigma_t[i - 1] = np.sqrt(np.mean(vels[mask, 1] ** 2)) / np.sqrt(2)

        return r, np.column_stack((sigma_r, sigma_t))






# -------------------------------------------------
# ------------ Profile fitting ------------------


def fit_sersic(I, r, init_vals):
    res = least_squares(ls_sersic, init_vals,
                        args=(r, I),
                        method='lm',
                        )

    return res.x


# [1e6, 100, 7, 0, r_e]
def fit_core_profile(I, r, init_vals):

    # mu_dot, r_b, alpha, gamma
    res = least_squares(ls_core_sersic, init_vals,
                        args=(r, I),
                        method='lm',
                        #bounds=([min(I), min(r), 0, 0], [max(I), max(r), 3, 0.3]),
                        )

    return res.x


def fit_nuker(I, r, init_vals):

    res = least_squares(ls_nuker, init_vals,
                        args=(r, I),
                        method='lm',
                        )

    return res.x

# Good core sersic initial values: [1e5, 300, 2, 0.01, r_e]
# Good nuker initial values: [1e6, 500, 0.7, 10, 0]

def fit_from_file(inflie, bins, init_pars, profile, exc=0, calc_re=True):

    data = np.genfromtxt(inflie)

    r = data[:, 0]
    I = data[:, 1]

    if exc != 0:
        r = r[:-exc]
        I = I[:-exc]

    if profile == 'c':

        if calc_re:
            r_e, I_e = find_effective_radius(data[:, 1], data[:, 0], bins)
            init_pars[4] = r_e

        return r, I, fit_core_profile(I, r, init_pars)

    elif profile == 'n':

        return r, I, fit_nuker(I, r, init_pars)

    elif profile == 's':

        if calc_re:
            r_e, I_e = find_effective_radius(data[:, 1], data[:, 0], bins)
            init_pars[1] = r_e
            init_pars[0] = I_e

        return r, I, fit_sersic(I, r, init_pars)

    else:

        return 0, 0, 0


# ----------------- Nonparametric method ----------

def polynomial(x, coeffs):
    y = np.zeros(x.size)
    expos = np.arange(len(coeffs))
    for i, coeff in enumerate(coeffs):
        y += x**expos[i]*coeff
    return y

def x_at_polynomial_value(x, coeffs, val=-1/2):
    return abs(polynomial(x, coeffs) - val)

def find_cusp_radius(r, I, init_guess, search_area, poly_deg=5):

    mask = r < search_area[1]
    mask *= r > search_area[0]

    I_grad = np.gradient(np.log10(I)[mask], np.log10(r)[mask])
    fit_coeffs = np.polyfit(r[mask], I_grad, poly_deg)
    fit_coeffs = np.flip(fit_coeffs)

    plt.plot(r[mask]/1000, polynomial(r[mask], fit_coeffs))
    plt.plot(r[mask]/1000, I_grad)
    plt.show()

    return minimize(x_at_polynomial_value, init_guess, args=(fit_coeffs), method='Nelder-Mead').x


# ------------------------------------------------

def read_coords_and_masses_from_ketju(file_num, partType, data_units):
    data = h5.File(filenames[file_num])
    coords = data[partType]['Coordinates'][:, :] * data_units.unit_length * unit_length_in_pc
    masses = data[partType]['Masses'][:] * data_units.unit_mass
    return coords, masses

def read_velocities_from_ketju(file_num, partType, data_units):
    data = h5.File(filenames[file_num])
    return data[partType]['Velocities'][:, :] * data_units.unit_vel * unit_vel_in_m_per_s / 1000

def print_half_mass_radius(file_nums):

    for num in file_nums:

        coords, masses = read_coords_and_masses_from_ketju(num, "PartType3", ketjugw.units.DataUnits())

        com = find_com(coords, masses, max(coords[:, 0]) * 0.001)
        coords = coords - com

        dists = np.sqrt(coords[:,0]**2 + coords[:,1]**2 + coords[:,2]**2)

        hm_r = np.sort(dists)[int(dists.size/2)]
        print("half mass radius:", hm_r)
        print("approximate effective radius:", hm_r * 3/4)


def print_influence_radius(file_nums):

    for num in file_nums:

        coords, masses = read_coords_and_masses_from_ketju(num, "PartType3", ketjugw.units.DataUnits())
        coords_bh, masses_bh = read_coords_and_masses_from_ketju(num, "PartType5", ketjugw.units.DataUnits())

        com = find_com(coords, masses, max(coords[:, 0]) * 0.001)
        coords = coords - com

        dists = np.sort(np.sqrt(coords[:,0]**2 + coords[:,1]**2 + coords[:,2]**2))

        r_m = np.sum(masses_bh)/masses[0]

        print(r_m)

        print("influence radius:", dists[int(r_m)]*3/4)



def plot_mus(filename_base, sim_nums):

    matplotlib.rcParams.update({"font.size":16})

    files = np.array([])
    for num in sim_nums:
        files = np.append(files, filename_base.format(num))

    for num, filename in zip(sim_nums, files):

        data = np.genfromtxt(filename)
        r = data[:,0]
        I = data[:,1]

        plt.plot(r/1000, I, color=colors[num], label='BH-{} Merger'.format(num), linewidth=3)


    plt.xlim(1e-2, 1e2)
    plt.ylim(10, 5e4)
    plt.semilogy()
    plt.semilogx()
    plt.xlabel("r [kpc]")
    plt.ylabel("$\mu$(r) [L$_\odot$pc$^{-2}$]")
    plt.legend(loc=3)
    plt.minorticks_on()
    plt.tick_params(right=True, top=True, which='both', direction='in')
    plt.show()


def plot_dots_and_com():
    data = h5.File(filenames[0])
    coordinates = data['PartType3']['Coordinates'][:,:]
    masses = data['PartType3']['Masses'][:]
    com = find_com(coordinates, masses, max(coordinates[:,0])*0.001)

    plt.plot(coordinates[:,0], coordinates[:,1], 'ob')
    plt.plot(com[0], com[1], 'or')
    plt.title("Final")
    plt.show()


def plot_core_sersic_profiles(infile, target_names, subplots=True, sp_r=2, sp_c=2):

    targets = np.genfromtxt(infile, usecols=(0,), skip_header=True, dtype=str)
    ps = np.genfromtxt(infile, skip_header=True)[:,1:]

    matplotlib.rcParams.update({"font.size":15})

    if subplots:

        for j, name in enumerate(target_names):
            i = np.where(targets == name)[0]
            x = np.logspace(0, 5, 500)

            I = core_sersic_profile(x, mu_in_lsun_per_pc_squared(4.83, ps[i, 0]), ps[i, 1],
                                    ps[i, 4], ps[i, 2], ps[i, 3], ps[i, 5])

            #parameter_text = "$r_b \\approx$ {:.2e}".format(ps[i, 1][0] / 1000)

            parameter_text = "$r_b =$ {:.2e}\n" \
                             "$\mu_b =$ {:.2f}\n" \
                             "$R_e =$ {:.4e}\n" \
                             "$n =$ {}\n" \
                             "$\\alpha =$ {:.2f}\n" \
                             "$\gamma =$ {:.2f}\n".format(ps[i,1][0] / 1000,
                                                          ps[i,0][0],
                                                          ps[i,2][0] / 1000, ps[i,5][0], ps[i,4][0], ps[i,3][0])

            plt.subplot(sp_r, sp_c, j+1)
            plt.plot(x / 1000, mu_in_mag_per_arcsec_squared(4.83, I), color='k')

            y_size = plt.ylim()[1] - plt.ylim()[0]
            plt.text(0.0015, plt.ylim()[1] - 0.015 * y_size, parameter_text)

            plt.title(name)
            plt.semilogx()
            plt.gca().invert_yaxis()
            plt.ylabel("$\mu_V$[mag arcsec$^{-2}$]")
            plt.xlabel("r[kpc]")


        plt.tight_layout()

    else:

        x = np.logspace(1, 5, 500)

        for name in target_names:

            i = np.where(targets == name)[0]

            I = core_sersic_profile(x, mu_in_lsun_per_pc_squared(4.83, ps[i, 0]), ps[i,1],
                                    ps[i,4], ps[i,2], ps[i,3], ps[i,5])

            # r, mu_b, r_b, alpha, r_e, gamma, n=4
            plt.plot(x / 1000,  mu_in_mag_per_arcsec_squared(4.83, I), label=name, linewidth=3)

        plt.xlim(min(x)/1000, max(x)/1000)
        plt.ylim(15, 26)

        plt.semilogx()
        plt.gca().invert_yaxis()
        plt.legend(loc=3)
        plt.ylabel("$\mu_V$[mag arcsec$^{-2}$]")
        plt.minorticks_on()
        plt.tick_params(which='both', direction='in', right=True, top=True)
        plt.xlabel("r[kpc]")

    plt.show()


def plot_and_show_mass_relations(infile):
    core_r = np.genfromtxt(infile, usecols=(0,))
    nuker_r = np.genfromtxt(infile, usecols=(2,))
    cusp_r = np.genfromtxt(infile, usecols=(1,))
    bh_masses = np.genfromtxt(infile, usecols=(3,), dtype=str)

    matplotlib.rcParams.update({'font.size': 14})

    plt.plot(core_r, color='k', marker='s', linestyle='', markersize=8, label="Core-Sérsic profile fit")
    plt.plot(nuker_r, color= 'b', marker='o', linestyle='', markersize=8, label="Nuker profile fit")
    plt.plot(cusp_r, color='r', marker='o', linestyle='', markersize=8, label="Non-parametric estimate")

    plt.xticks(range(6), bh_masses)

    plt.ylim(0, 1)

    plt.xlabel("M$_\\bullet$ [M$_\odot$]")
    plt.ylabel("r$_b$ [kpc]")

    plt.legend()

    plt.show()


def plot_sersic_fits(r, I, pars, a0, a1):

    # Initialize theoretical plot

    x = np.logspace(np.log10(min(r)), np.log10(max(r)), 500)
    y_core = mu_in_mag_per_arcsec_squared(4.83, sersic_profile(x, pars[0], pars[1]))

    I = mu_in_mag_per_arcsec_squared(4.83, I)

    # pars0 = mu, pars1 = r_b, pars2 = alpha, pars3 = gamma, pars4 = r_e, pars5 = n
    parameter_text = "$R_e =$ {:.4e}\n" \
                     "$I_e =$ {:.4e}\n" \
                     "$n =$ {:.1f}\n".format(pars[1] / 1000, mu_in_mag_per_arcsec_squared(4.83, pars[0]), pars[2])

    print(parameter_text)

    rs = I - mu_in_mag_per_arcsec_squared(4.83, sersic_profile(r, pars[0], pars[1]))
    rms = np.sqrt(np.mean(rs ** 2))

    # Plotting

    # a0.set_title("Core-Sérsic profile fit \n $M_{\\bullet} = 1.7 \\times 10^9 M_{\odot}$")

    a0.plot(r / 1000, I, marker='o', markersize=4, label='Calculated profile')
    a0.plot(x / 1000, y_core, '-g', linewidth=2, label="Sérsic profile")
    a0.set_xlim(1e-2, 1e2)
    a0.semilogx()
    a0.set_ylabel("$\mu_V$(r) [mag arcsec$^{-2}$]")
    y_size = a0.get_ybound()[1] - a0.get_ybound()[0]
    a0.text(0.02, a0.get_ybound()[1] - 0.2 * y_size, parameter_text, fontsize=12)
    # a0.text(4, a0.get_ybound()[1]-0.9*y_size, "Core-Sérsic", fontsize=12)
    a0.legend(loc=3)
    a0.invert_yaxis()
    a0.tick_params(direction='in', bottom=False, right=True, top=True)  # , labelbottom=False)

    a1.axhline(color='k')
    a1.plot(r / 1000, rs, 'ob', markersize=3)
    a1.set_xlim(1e-2, 1e2)
    a1.semilogx()
    a1.set_ylabel("Residuals")
    a1.invert_yaxis()
    y_size = a1.get_ybound()[1] - a1.get_ybound()[0]
    a1.text(0.015, a1.get_ybound()[0] - 0.05 * y_size, "$\Delta =$ {:.4f}".format(rms))



def plot_core_fits(r, I, pars, a0, a1):

    # Initialize theoretical plot

    x = np.logspace(np.log10(min(r)), np.log10(max(r)), 500)
    y_core = mu_in_mag_per_arcsec_squared(4.83, core_sersic_profile(x, pars[0], pars[1], pars[2], pars[4], pars[3]))

    I = mu_in_mag_per_arcsec_squared(4.83, I)

    # pars0 = mu, pars1 = r_b, pars2 = alpha, pars3 = gamma, pars4 = r_e, pars5 = n
    parameter_text = "$r_b =$ {:.4e}\n" \
                     "$\mu_b =$ {:.4f}\n" \
                     "$R_e =$ {:.4e}\n" \
                     "$n =$ {}\n" \
                     "$\\alpha =$ {:.4f}\n" \
                     "$\gamma =$ {:.4f}\n".format(pars[1]/1000, mu_in_mag_per_arcsec_squared(4.83, pars[0]),
                                                  pars[4]/1000, 4, pars[2], pars[3])

    print(parameter_text)

    rs = I - mu_in_mag_per_arcsec_squared(4.83, core_sersic_profile(r, pars[0], pars[1], pars[2], pars[4], pars[3]))
    rms = np.sqrt(np.mean(rs**2))

    # Plotting

    #a0.set_title("Core-Sérsic profile fit \n $M_{\\bullet} = 1.7 \\times 10^9 M_{\odot}$")


    a0.plot(r / 1000, I, marker='o', markersize=4, label='Simulated data')
    a0.plot(x / 1000, y_core, '-g', linewidth=2, label="Core-Sérsic profile fit")
    a0.set_xlim(1e-2, 1e2)
    a0.semilogx()
    a0.set_ylabel("$\mu_V$(r) [mag arcsec$^{-2}$]", fontsize=14)
    y_size = a0.get_ybound()[1] - a0.get_ybound()[0]
    a0.text(0.02, a0.get_ybound()[1]-0.2*y_size, parameter_text, fontsize=16)
    #a0.text(0.05, 0.05, parameter_text, fontsize=14, transform=a0.transAxes)
    a0.text(4, a0.get_ybound()[1]-0.9*y_size, "Core-Sérsic", fontsize=16)
    a0.legend(loc=3)
    a0.invert_yaxis()
    a0.tick_params(direction='in', bottom=False, right=True, top=True)#, labelbottom=False)


    a1.axhline(color='k')
    a1.plot(r/1000, rs, 'ob', markersize=3)
    a1.set_xlim(1e-2, 1e2)
    a1.semilogx()
    a1.set_ylabel("Residuals", fontsize=14)
    a1.invert_yaxis()
    y_size = a1.get_ybound()[1] - a1.get_ybound()[0]
    a1.text(0.015, a1.get_ybound()[0]-0.05*y_size, "$\Delta =$ {:.4f}".format(rms), fontsize=14)
    a1.set_xlabel("r [kpc]", fontsize=14)


def plot_nuker_fits(r, I, pars, a0, a1):

    # Initialize theoretical plot

    x = np.logspace(np.log10(min(r)), np.log10(max(r)), 500)
    y = mu_in_mag_per_arcsec_squared(4.83, nuker_profile(x, pars[0], pars[1], pars[2], pars[3], pars[4]))

    I = mu_in_mag_per_arcsec_squared(4.83, I)

    # pars0 = mub, pars1 = rb, pars2 = beta, pars3 = alpha, pars4 = gamma
    parameter_text = "$r_b =$ {:.4e}\n" \
                     "$\mu_b =$ {:.4f}\n" \
                     "$\\alpha =$ {:.4f}\n" \
                     "$\\beta =$ {:.4f}\n" \
                     "$\gamma =$ {:.4f}\n".format(pars[1]/1000, mu_in_mag_per_arcsec_squared(4.38, pars[0]),
                                                  pars[3], pars[2], (pars[4]))

    print(parameter_text)

    rs = I - mu_in_mag_per_arcsec_squared(4.83, nuker_profile(r, pars[0], pars[1], pars[2], pars[3], pars[4]))
    rms = np.sqrt(np.mean(rs ** 2))

    # Plotting

    #a0.set_title("Nuker profile fit \n $M_{\\bullet} =  1.7 \\times 10^9 M_{\odot}$")


    a0.plot(r / 1000, I, marker='o', markersize=4, label='Simulated data')
    a0.plot(x / 1000, y, '-g', linewidth=2, label="Nuker profile fit")
    a0.set_xlim(1e-2, np.log10(3100))
    a0.semilogx()
    a0.set_ylabel("$\mu_V$(r) [mag arcsec$^{-2}$]", fontsize=14)
    y_size = a0.get_ybound()[1] - a0.get_ybound()[0]
    a0.text(0.015, a0.get_ybound()[1] - 0.2 * y_size, parameter_text, fontsize=16)
    #a0.text(0.05, 0.05, parameter_text, fontsize=14, transform=a0.transAxes)
    a0.text(1, a0.get_ybound()[1] - 0.9 * y_size, "Nuker", fontsize=16)
    a0.invert_yaxis()
    a0.legend(loc=3)
    a0.tick_params(direction='in', bottom=False, right=True, top=True)#, labelbottom=False)


    a1.axhline(color='k')
    a1.plot(r/1000, rs, 'ob', markersize=3)
    a1.set_xlim(1e-2, np.log10(3100))
    a1.semilogx()
    a1.set_ylabel("Residuals", fontsize=14)
    a1.invert_yaxis()
    y_size = a1.get_ybound()[1] - a1.get_ybound()[0]
    a1.text(0.015, a1.get_ybound()[0] - 0.05 * y_size, "$\Delta =$ {:.4f}".format(rms), fontsize=14)
    a1.set_xlabel("r [kpc]", fontsize=14)


def plot_and_show_nuker_and_core(r_c, I_c, r_n, I_n, core_pars, nuker_pars):

    matplotlib.rcParams.update({'font.size': 14})

    f, ((a0, a2), (a1, a3)) = plt.subplots(2, 2, gridspec_kw={'height_ratios': [4, 1]}, constrained_layout=True)

    plot_core_fits(r_c, I_c, core_pars, a0, a1)
    plot_nuker_fits(r_n, I_n, nuker_pars, a2, a3)

    #plt.tight_layout(pad=0, rect=(0, 0.05, 0.95, 0.95))

    plt.show()


# s: smoothing
def plot_and_show_beta(file_nums, bins, s, use_r_b=False):

    matplotlib.rcParams.update({'font.size': 14})

    r_b = 0
    if use_r_b:

        used_targets = np.array([])
        for i in range(len(file_nums)):
            used_targets = np.append(used_targets, ['BH-' + str(file_nums[i]) + '_Merger'])

        targets = np.genfromtxt('core_sersic_profiles.dat', usecols=(0,), skip_header=True, dtype=str)
        ps = np.genfromtxt('core_sersic_profiles.dat', skip_header=True)[:, 1:]

        mask = np.in1d(targets, used_targets)
        r_b = ps[mask, 1]


    for i, num in enumerate(file_nums):

        coords, masses = read_coords_and_masses_from_ketju(num, 'PartType3', ketjugw.units.DataUnits())
        vels = read_velocities_from_ketju(num, 'PartType3', ketjugw.units.DataUnits())

        r, sigmas = velocity_dispersion_profile(coords, masses, vels, bins, method='p')
        beta = 1 - (sigmas[:, 1] ** 2) / (sigmas[:, 0] ** 2)

        if use_r_b:
            #r = bins[:-1] / r_b[i]
            r = r / r_b[i]
        else:
            #r = bins[:-1] / 1000
            r = r / 1000

        # Smooth betas after r=1
        #mask = r > 1
        #beta[mask] = smooth(beta[mask], 2)

        if s == 0:
            print(r)
            plt.plot(r, beta, color=colors[num], label='BH-{} merger'.format(num), linewidth=2)
        else:
            plt.plot(r, smooth(beta, s), color=colors[num], label='BH-{} merger'.format(num), linewidth=2)

    plt.semilogx()
    plt.xlim(0.3, 30)       # 0.3 - 15
    plt.ylim(-0.8, 0.7)     # -1.2 - 0.6
    if use_r_b:
        plt.xlabel("$r/r_b$")
    else:
        plt.xlabel("r[kpc]")
    plt.ylabel("$\\beta$(r)")
    plt.minorticks_on()
    plt.tick_params(right='True', top='true', direction='in', which='both', length=5)
    plt.tick_params(which='major', length=10)
    #plt.tick_params(right=False, left=False, which='minor')
    plt.xticks([1, 10])
    plt.yticks([-0.5, 0, 0.5])
    plt.legend(loc=4)
    plt.show()


def plot_all_core_nuker_fits(infiles, bins, exc=0, profile='c'):

    f, ax = plt.subplots(6, 2, figsize=(15, 20), gridspec_kw={'height_ratios': [4, 1, 4, 1, 4, 1]})

    init_pars = [0]
    if profile == 'c':
        init_pars = [1e5, 300, 2, 0.01, 0]
    elif profile == 'n':
        init_pars = [1e6, 500, 0.7, 10, 0]

    identifiers = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # Good core sersic initial values: [1e5, 300, 2, 0.01, r_e]
    # Good nuker initial values: [1e6, 500, 0.7, 10, 0]

    for i, file in enumerate(infiles):

        r, I, pars = fit_from_file(file, bins, init_pars, profile=profile, exc=exc, calc_re=True)

        x = i%2
        y1 = int(np.floor(i/2)*2)
        y2 = int(np.floor(i/2)*2+1)

        a0 = ax[y1, x]
        a1 = ax[y2, x]

        if profile == 'c':
            plot_core_fits(r, I, pars, a0, a1)
        if profile == 'n':
            plot_nuker_fits(r, I, pars, a0, a1)

        # Remove labels from problematic places:
        if(i % 2 != 0):
            a0.set_ylabel('')
            a1.set_ylabel('')

        if(i != 4 and i != 5):
            a1.set_xlabel('')

        a0.tick_params(labelsize=14)
        a1.tick_params(labelsize=14)

        a0.text(0.9, 0.9, identifiers[i], transform=a0.transAxes, fontsize=20)



    plt.savefig('dummy.png')





def smooth(xs, m):
    res = np.zeros(len(xs))
    for i in range(0, m):
        res[i] = np.mean(xs[:i+m])
    for i in range(m, len(res)-m):
        res[i] = np.mean(xs[i-m:i+m])
    for i in range(len(res)-m, len(res)):
        res[i] = np.mean(xs[i-m:])
    return res


def write_data_to_file(outfile, file_num, bins):

    coords, masses = read_coords_and_masses_from_ketju(file_num, 'PartType3', ketjugw.units.DataUnits())

    densities, r = mean_surface_density_profile(coords, masses, bins, 100)
    I = densities * 1 / 4

    outdata = np.column_stack((r, I))

    np.savetxt(outfile, outdata)


# ------- Some maybe usable stuff ------------

def smooth(xs, m):
    print("smoothing")
    res = xs
    for i in range(m, len(res)-m):
        res[i] = np.mean(xs[i-m:i+1+m])
    return res

def thin_smooth(data, c):

    n = 2*c+1
    res = np.zeros(int(len(data)/n))

    for i in range(len(res)):
        res[i] = np.mean(data[i*n:(i+1)*n])

    return res

def surface_density_profile_binned_by_particle_num(coords, masses, parts_in_bin):

    com = find_com(coords, masses,  max(coords[:,0])*0.001)
    coords -= com

    dists = np.sort(np.sqrt(coords[:,0]**2 + coords[:,1]**2))

    N = int(dists.size / parts_in_bin)

    r = np.zeros(N)
    rho = np.zeros(N)

    print(N)

    for i in range(0, N):

        r[i] = (dists[i*parts_in_bin] + dists[(i+1)*parts_in_bin-1]) / 2
        rho[i] = parts_in_bin*masses[i] / (-np.pi * dists[i*parts_in_bin]**2 + np.pi * dists[(i+1)*parts_in_bin-1]**2)

    return r, rho


# -------------------------------------------



def main():


    coords, masses = read_coords_and_masses_from_ketju(3, 'PartType3', ketjugw.units.DataUnits())
    r, rho = surface_density_profile_binned_by_particle_num(coords, masses, 1000)

    c = 100
    r = np.concatenate((np.array(r[0:3]), thin_smooth(r, c)))
    rho = np.concatenate((np.array(rho[0:3]), thin_smooth(rho, c)))
    I = rho/4

    print(len(r))

    np.savetxt("particlebins_test.dat", np.column_stack((r, I)))

    r, I, pars = fit_from_file("particlebins_test.dat", basic_bins20, [1e5, 300, 2, 0, 1e8], 'c', calc_re=False)

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
    plot_core_fits(r, I, pars, a0, a1)
    plt.show()


    '''
    files = [
        "file_1_50bins.dat",
        "file_2_50bins.dat",
        "file_3_50bins.dat",
        "file_4_50bins.dat",
        "file_5_50bins.dat",
        "file_6_50bins.dat"
        ]

    nuker_files = [
        "nuker_file_1_50.dat",
        "nuker_file_2_50.dat",
        "nuker_file_3_50.dat",
        "nuker_file_4_50.dat",
        "nuker_file_5_50.dat",
        "nuker_file_6_50.dat"
        ]


    r = np.genfromtxt('100_bin_100_mean_BH6.dat', usecols=(0,))
    I = np.genfromtxt('100_bin_100_mean_BH6.dat', usecols=(1,))


    print(find_cusp_radius(r, I, 5e3, [200, 800]))
    '''

    '''
    r, I, pars = fit_from_file('file_0_20bins.dat', basic_bins20, [0, 0, 4], 's', calc_re=True, exc=1)
    
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
    plot_sersic_fits(r, I, pars, a0, a1)
    plt.show()
    '''

    #plot_mus('100_bin_100_mean_BH{}.dat', np.arange(0, 7))

    # beta_bins_rb
    #plot_and_show_beta(np.arange(2, 3), beta_bins, 2, use_r_b=True)
    #plot_and_show_beta(np.arange(0, 7), beta_bins, 2, use_r_b=False)


    #plot_core_sersic_profiles('core_sersic_profiles.dat', ['BH-6_Merger', 'NGC_1600'], subplots=False)
    #plot_core_sersic_profiles('core_sersic_profiles.dat', ['BH-2_Merger',  'NGC_4472'], subplots=False)

    #print_half_mass_radius([0, 1, 2, 3, 4, 5, 6])
    #print_influence_radius([1, 2, 3, 4, 5, 6])

    '''
    # Good core sersic initial values: [1e5, 300, 2, 0.01, r_e]
    # Good nuker initial values: [1e6, 500, 0.7, 10, 0]
    
    r_c = np.genfromtxt("file_3_50bins.dat", usecols=(0,))[:-4]
    I_c = np.genfromtxt("file_3_50bins.dat", usecols=(1,))[:-4]
    
    r_n = np.genfromtxt("nuker_file_3_50.dat", usecols=(0,))
    I_n = np.genfromtxt("nuker_file_3_50.dat", usecols=(1,))
    
    core_pars = fit_core_profile(I_c, r_c, [1e5, 300, 2, 0.01, 1e4])
    nuker_pars = fit_nuker(I_n, r_n, [1e6, 500, 0.7, 10, 0])
    
    plot_and_show_nuker_and_core(r_c, I_c, r_n, I_n, core_pars, nuker_pars)
    '''


if __name__ == "__main__":
    main()
