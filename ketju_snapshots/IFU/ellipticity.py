import pygad
import voronoimapspygad as vm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import minimize


# -------------- Useless junk ------------------

def second_moments(x, y, r, m):
    m_xx = np.average(x**2 / r**2, weights=m)
    m_yy = np.average(y**2 / r**2, weights=m)
    m_xy = np.average(x*y / r**2, weights=m)
    return m_xx, m_yy, m_xy


def mass_grid(x, y, m, extent, N):

    delta = extent / (2*N+1)
    grid = np.zeros((N, N))

    for i in range(-N, N+1):

        print i

        for j in range(-N, N+1):

            mask = (i*delta < x) * ((i+1) * delta > x) * (j*delta < y) * ((j+1) * delta > y)
            grid[i, j] = np.sum(m[mask])

    return grid

def density_around_point(x, y, coords, masses, delta):
    mask = (x - delta < coords[:,0]) * (x + delta > coords[:,0]) *  (y - delta < coords[:,1]) * (y + delta > coords[:,1])
    return np.sum(masses[mask])


def mass_inside_ellipse(a, b, coords, masses):
    inds = coords[:,0]**2 / a**2 + coords[:,1]**2 / b**2 <= 1
    return np.sum(masses[inds])

def half_mass_residual(pars, coords, masses, half_mass):
    y_dens = density_around_point(0, pars[1]/2, coords, masses, 5e-3)
    x_dens = density_around_point(pars[0]/2, 0, coords, masses, 5e-3)
    print abs(half_mass - mass_inside_ellipse(pars[0], pars[1], coords, masses)), abs(y_dens - x_dens)
    return abs(half_mass - mass_inside_ellipse(pars[0], pars[1], coords, masses)) + abs(y_dens - x_dens)


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


def get_half_mass(snap):
    return np.sum(snap['mass'][:])/2

# -----------------------------------------


def calc_shape_tensor(coords, masses, r_ell, b_per_a, c_per_a):

    mask = np.sqrt( coords[:,0]**2 + coords[:,1]**2/b_per_a**2 +  coords[:,2]**2/c_per_a**2) < r_ell

    M_tot = np.sum(masses[mask])

    masses = np.column_stack((masses[mask], masses[mask], masses[mask]))

    M = np.matmul(np.transpose(masses * coords[mask]), coords[mask])


    return M/M_tot

def iterate_shape_tensors(coords, masses, r_ell):

    S_init = calc_shape_tensor(coords, masses, r_ell, 1, 1)

    eig_vals, eig_vects = np.linalg.eig(S_init)

    eig_vals = np.sort(eig_vals)

    ba = np.sqrt(eig_vals[0] / eig_vals[2])
    ca = np.sqrt(eig_vals[1] / eig_vals[2])

    while 1:

        ba_0 = ba
        ca_0 = ca

        S = calc_shape_tensor(coords, masses, r_ell, ba_0, ca_0)

        eig_vals, eig_vects = np.linalg.eig(S)

        eig_vals = np.sort(eig_vals)

        print eig_vals

        ba = np.sqrt(eig_vals[0] / eig_vals[2])
        ca = np.sqrt(eig_vals[1] / eig_vals[2])

        print abs(ba-ba_0)/ba_0, abs(ca-ca_0)/ca_0

        if abs(ba-ba_0)/ba_0 <= 1e-3 and abs(ca-ca_0)/ca_0 <= 1e-3:
            break

    return ba, ca
    
    


filename = '../gamma-1.5-bh-6.hdf5'

r_e = 10.722


s = pygad.Snap(filename)
snap = s.stars

com = find_com(snap['pos'][:,:], snap['mass'][:], max(snap['pos'][:,0]) * 0.001)
snap['pos'][:,:] = snap['pos'][:,:] - com

snap = vm._orientsnap(snap, axisorientation=1, rangmom=0.5*20)
snap.to_physical_units()

print(np.sum(snap['mass'][:]))

mask = snap['pos'][:,0]**2 + snap['pos'][:,1]**2 + snap['pos'][:,2]**2 < r_e**2


coords = snap['pos'][mask,:]
masses = snap['mass'][mask]


#mask = np.random.choice(np.ones( len(masses) ), len(masses)/2)

#coords = snap['pos'][mask,:]
#masses = snap['mass'][mask]


ba, ca = iterate_shape_tensors(coords, masses, r_e)

print '$\epsilon_e =$', 1-ba
print ba, ca


#print(mass_grid(x, y, m, 20, 100))

'''
m_yy, m_xx, m_yx = second_moments(snap['pos'][:,0], snap['pos'][:,1], snap['rcyl'], snap['mass'][:])

q = m_xx - m_yy

print 1 - (1-q) / (q+1)

q = m_yx

print 1 - (1-q) / (q+1)

'''

'''
hm = get_half_mass(snap)
print hm
init_vals = [10, 6]
pars = minimize(half_mass_residual, init_vals, args=(snap['pos'][:,:], snap['mass'][:], hm), method='Powell')
print pars.x
'''
