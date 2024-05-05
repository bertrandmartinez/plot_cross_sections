import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, mu_0, N_A, k, h, hbar, alpha, value

# Classical electron radius
r_e = value('classical electron radius')

def read_data(Z, T1):

    # Read the data copy/pasted from S.M. Seltzer and M.J. BergerNuclear Instruments and Methods in Physics Research B12 (1985) 95-134

    # Define size of arrays
    Nx = 14
    Ny = 31
    cs_absx = np.zeros(Nx)
    cs_absy, cs_tot, cs_rad = np.zeros(Ny), np.zeros(Ny), np.zeros(Ny)
    cs_seltzer = [np.zeros(Nx) for n1 in range(Ny)]

    # open the file
    f = open('../../utils/Seltzer_Berger_Data_Z{:2d}.txt'.format(Z),'r')
    data  = f.readlines()
    f.close()
    N = len(data)
    line = data[1].split()

    # axis with the energy of the incident electron
    for i in range(Nx):
        cs_absx[i] = float(line[i])

    # array with the differential cross-section
    step = 4
    for i in range(step,step+Ny):
        line = data[i].split()
        j = int(i-step)
        cs_absy[j] = float(line[0])
        cs_tot[j] = float(line[Nx])
        cs_rad[j] = float(line[Nx+1])
        
        # Convert in ISU m^2
        for k in range(Nx):
            beta2 = 1.-1./(1. + cs_absy[j] / 0.511)**2
            fac = (Z**2/beta2) * 1.e-31
            cs_seltzer[j][k] = float(line[k+1]) * fac

    # Interpolate for the electron energy
    index = abs(cs_absy-1.e-3*T1).argmin()

    return cs_absx, cs_seltzer[index]

def tot_cs_SB(Z):
# Do not use this routine, it is still experimental and not validated

    # Compute the total cross-section from the data copy/pasted from S.M. Seltzer and M.J. BergerNuclear Instruments and Methods in Physics Research B12 (1985) 95-134

    # Define size of arrays
    M, N = 31, 14
    axis_k = np.zeros(N)
    axis_g1, phi, sigma = np.zeros(M), np.zeros(M), np.zeros(M)
    cs_seltzer = np.zeros([M, N])

    # open the file
    f = open('../../utils/Seltzer_Berger_Data_Z{:2d}.txt'.format(Z),'r')
    data  = f.readlines()
    f.close()
    L = len(data)
    line = data[1].split()

    # axis with the energy of the photon
    for k in range(N):
        axis_k[k] = float(line[k])

    # array with the differential cross-section
    step = 4
    for i in range(step,step+M):

        # Read the line
        line = data[i].split()
        j = int(i-step)

        # Energy of incident electron (MeV)
        axis_g1[j] = float(line[0])
        
        # Dfferential cross-section
        for k in range(N):
            cs_seltzer[j][k] = float(line[k+1])

        # Radiative cross-section
        phi[j] = float(line[N+1])

        # convert to ISU
        b2 = 1.-1./(1. + axis_g1[j] / 0.511)**2
        fac = (Z**2/b2) * 1.e-31
        cs_seltzer[j] *= fac

    sigma = np.zeros(M)
    phirad = np.zeros(M)
    for j in range(M):
        tmp = axis_k * (axis_g1[j] - 1.0)
        for k in range(1,N):
            sigma[j] += cs_seltzer[j][k]  * (axis_k[k] - axis_k[k-1]) / axis_k[k]
            phirad[j] += cs_seltzer[j][k] * (axis_k[k] - axis_k[k-1])

    # convert phi
    phi *= 4. * alpha * (r_e * Z)**2 * axis_g1 * m_e * c**2# / (4.*np.pi*epsilon_0)
    print(phi/phirad)
    print(phirad/phi)

    return axis_g1, sigma, phirad

