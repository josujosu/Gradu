import os
import pygad
import voronoimapspygad as vm
import numpy as np


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

#com = find_com(coords, masses, max(coords[:, 0]) * 0.001)


hdf5directorystart = "../gamma-1.5-bh-"

hdf5directoryend   = ".hdf5"

name_start = 'BH_'
name_end   = '.png'

for file in range(0,7):

    filename = hdf5directorystart + str(file) + hdf5directoryend
    print filename
    output = name_start+str(file)
    print filename

    s=pygad.Snap(filename)
    s.to_physical_units()
    snap=s.stars

    com = find_com(s['pos'][:,:], s['mass'][:], max(s['pos'][:,0]) * 0.001)
    s['pos'][:,:] = s['pos'][:,:] - com
    
    print s['pos'].units
    print s['vel'].units

    #posx = np.copy(s["pos"][:,0])
    #posy = np.copy(s["pos"][:,1])
    #velx = np.copy(s["vel"][:,0])
    #vely = np.copy(s["vel"][:,1])

    
    vm.voronoimap(snap,'vel',sigmapp=-0.2, weightqty='mass', npseudoparticles=10,npixel_per_side=99 ,extent=20, nspaxels=1000, plotfile=output, savetxt="txt-nspaxels_2000.txt", cmaplimits=[ [-20, 20], [270, 370], [-0.1, 0.1], [-0.1, 0.1] ], addlambdar=True, figureconfig='22')
