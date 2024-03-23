import numpy as np
from qed_rates import factor_Elwert, nr_cs_dif
import matplotlib.pyplot as plt
plt.style.use('prl.style')


# physical parameters
Z = 29
T1 = 100.
gamma_1 = 1. + T1 / 511.

# discretization of k axis
Nk = 100
axis_k  = np.linspace(0., gamma_1-1., Nk)

# screened cross-section from Fig.2 after correction
elwert = np.array([ factor_Elwert(Z, k, gamma_1) for k in axis_k ])

cs_screen_elwert_on = np.array([ nr_cs_dif(Z, k, gamma_1, True) for k in axis_k ])
cs_screen_elwert_on = axis_k * cs_screen_elwert_on

cs_screen_elwert_no = np.array([ nr_cs_dif(Z, k, gamma_1, False) for k in axis_k ])
cs_screen_elwert_no = axis_k * cs_screen_elwert_no

#-----------------------------------------------------------------------
# Section efficace en energie selon Seltzer
#-----------------------------------------------------------------------
Nx = 14
Ny = 31
cs_absx = np.zeros(Nx)
cs_absy, cs_tot, cs_rad = np.zeros(Ny), np.zeros(Ny), np.zeros(Ny)
cs_seltze = [np.zeros(Nx) for n1 in range(Ny)]
f = open('seltzerZ29.txt','r')
data  = f.readlines()
f.close()
N = len(data)
line = data[1].split()
for i in range(Nx):
  cs_absx[i] = float(line[i])
step = 4
for i in range(step,step+Ny):
  line = data[i].split()
  j = int(i-step)
  cs_absy[j] = float(line[0])
  cs_tot[j] = float(line[Nx])
  cs_rad[j] = float(line[Nx+1])
  for k in range(Nx):
    beta2 = 1.-1./(1. + cs_absy[j] / 0.511)**2
    fac = (Z**2/beta2) * 1.e-31
    cs_seltze[j][k] = float(line[k+1]) * fac

#-----------------------------------------------------------------------
# Figure
#-----------------------------------------------------------------------

fig, axs = plt.subplots(1, 1, figsize=(6,4))
axs.plot(cs_absx,cs_seltze[6], c='r', label=r"$\sigma_{SB}$")
axs.plot(axis_k[1:-1] / (gamma_1 - 1.), cs_screen_elwert_on[1:-1], c='k', label="Elwert")
axs.plot(axis_k[1:-1] / (gamma_1 - 1.), cs_screen_elwert_no[1:-1], c='k', ls='--', label="no Elwert")
axs.set_xlabel(r"$ k / (\gamma_1 - 1) $")
axs.set_xlim([0.,1.])
axs.set_ylabel(r"$ k d \sigma / dk \, \rm (m^2)$")
axs.set_ylim([0.,8.e-27])
location = np.linspace(0., 8.e-27, 9)
axs.yaxis.set_ticks(location)
axs.yaxis.set_ticklabels([r'${0:.{1}f}$'.format(1.e27*elem, 0) for elem in location])
axs.text(0.08, 1.1, r'$ \times 10^{-27} $', ha='center', va='center', transform=axs.transAxes, bbox=dict(alpha=0, facecolor="white", edgecolor="white") )
axs.legend(loc="best")
plt.tight_layout()
plt.savefig("Elwert_factor.png", dpi=150)

fig, axs = plt.subplots(1, 1, figsize=(6,4))
axs.plot(axis_k / (gamma_1 - 1.), elwert, c='r')
axs.set_xlabel(r"$ k / (\gamma_1 - 1) $")
axs.set_ylabel(r"$ f $")
plt.tight_layout()
plt.show()