import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def w2(u):
    if u < 0:
        return 0
    if u < 0.5:
        return 16/3*u**2 - 48/5*u**4 + 32/5 * u**5 - 14/5
    if u >= 1:
        return -1/u
    else:
        return 1/15*u**-1 + 32/3 * u**2 - 16*u**3 + 48/5 * u**4 - 32/15 * u**5 - 16/5

# Assumes that epsilon=1
def splineKernelSoftening(r,  G=1, M=1, h=2.8):

    w = np.zeros(len(r))

    for i in range(0, w.size):
        w[i] = w2(r[i]/h)

    return G*M/h * w


def plummerSoftening(r, G=1, M=1, epsilon=1):
    return - G*M / np.sqrt(r**2 + epsilon**2)

def pureNewtonian(r, G=1, M=1):
    return -G*M/r


r = np.linspace(0, 6, 1000)
newton = pureNewtonian(r)
plummer = plummerSoftening(r)
spline = splineKernelSoftening(r)

matplotlib.rcParams.update({"font.size": 16})

plt.axvline(2.8, linestyle='--', linewidth=3, color='k')
plt.plot(r, newton, linewidth=3, label='Newtonian potential')
plt.plot(r, plummer, linewidth=3, label='Plummer softened potential')
plt.plot(r, spline, linewidth=3, label='Spline-kernel softened potential')
plt.text(2.9, -1.4, '$h_\mathrm{ML} = 2.8 \epsilon$', fontsize=20)
plt.xlabel('r')
plt.ylabel('$\phi$ / $|\phi_\mathrm{Plummer}(0)|$')
plt.ylim(-1.5, 0)
plt.xlim(0, 4)
plt.legend(loc=2)
plt.tick_params(right=True, top=True, which='both', direction='in')
plt.show()
