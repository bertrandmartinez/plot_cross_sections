import numpy as np

def read_data(Z):
    Nx = 14
    Ny = 31
    cs_absx = np.zeros(Nx)
    cs_absy, cs_tot, cs_rad = np.zeros(Ny), np.zeros(Ny), np.zeros(Ny)
    cs_seltzer = [np.zeros(Nx) for n1 in range(Ny)]
    f = open('../../utils/Seltzer_Berger_Data_Z{:2d}.txt'.format(Z),'r')
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
        cs_seltzer[j][k] = float(line[k+1]) * fac
    return cs_absx, cs_seltzer
