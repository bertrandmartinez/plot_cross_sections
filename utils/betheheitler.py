from scipy.special import zeta
from scipy.constants import c, e, m_e, epsilon_0, mu_0, N_A, k, h, hbar, alpha, value
from scipy.integrate import quad
from screening import Fermi_length, Debye_length, Coulomb_correction, I1_func, I2_func, reduced_potential_length

# Classical electron radius
r_e = value('classical electron radius')

def bh_dif_cs_sp(Z, gp, k, L):
    '''
    Bethe Heitler differential cross-section assuming screening with a single-exponential potential (Fermi)
    Eq.(25) from B. Martinez et al, Physics of Plasmas, 36, 103109 (2019)
    inputs :
        Z : atomic number
        gp : Lorentz factor of the emitted positron
        k : normalized energy of the photon k = hbar \omega / mc^2
        L : Fermi length normalized by Compton wavelength
    outputs :
        result : differential cross-section  (m^2)
    '''

    result = 0.0
    condition = (k >= 2.) and (gp >= 1.) and (gp <= k-1.)

    if condition:

        eta = 1.0
        ge = k - gp
        d = k / (2.0 * gp * ge)

        # Coulomb correction term
        fc = 0.0
        if k > 200. :
            fc = Coulomb_correction(Z)

        T1 = 4. * (Z * r_e) ** 2 * alpha / k ** 3
        T2 = (gp ** 2 + ge ** 2) * (I1_func(d, L, eta) + 1.0 - fc)
        T3 = (2. / 3.) * gp * ge * (I2_func(d, L, eta) + 5. / 6. - fc)

        result = T1 * (T2 + T3)

    return result

def bh_dif_cs_dp(Z, gp, k, T, ni, Zstar):
    '''
    Bethe Heitler differential cross-section assuming screening with a double-exponential potential (Fermi + Debye)
    Eq.(25) from B. Martinez et al, Physics of Plasmas, 36, 103109 (2019)
    inputs :
        Z : atomic number
        gp : Lorentz factor of the emitted positron
        k : normalized energy of the photon k = hbar \omega / mc^2
        T : temperature of plasma (keV)
        ni : density of plasma (/m3)
        Zstar : ionization degree of plasma
    outputs :
        result : differential cross-section  (m^2)
    '''

    result = 0.0
    condition = (k >= 2.) and (gp >= 1.) and (gp <= k-1.)

    if condition:

        eta = 1.0
        ge = k - gp
        d = k / (2.0 * gp * ge)

        # Determine the scale-length of the reduced potential
        etaf = 1. - Zstar / Z
        etad = 1. - etaf
        
        Lf = Fermi_length(Z)
        Ld = Debye_length(T, ni, Zstar)  # Debye Length

        # Eq. (22)
        etar = 1.0
        Lr = reduced_potential_length(etaf, Lf, etad, Ld)
        
        # Coulomb correction term
        fc = 0.0
        if k > 200. :
            fc = Coulomb_correction(Z)

        T1 = 4. * (Z * r_e) ** 2 * alpha / k ** 3
        T2 = (gp ** 2 + ge ** 2) * (I1_func(d, Lr, etar) + 1.0 - fc)
        T3 = (2. / 3.) * gp * ge * (I2_func(d, Lr, etar) + 5. / 6. - fc)

        result = T1 * (T2 + T3)

    return result

def bh_tot_cs_sp(Z, k, L):
    '''
    Bethe-Heitler total cross-section assuming screening with a single-exponential potential (Fermi)
    Integration of Eq.(25) from B. Martinez et al, Physics of Plasmas, 36, 103109 (2019)
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        L : Fermi length normalized by Compton wavelength
    outputs :
        result : total cross-section  (m^2)
    '''

    condition = (k >= 2.)

    result = 0.0
    if condition:
        # we switch the order of variables for quad
        bh_dif_cs_sp_reorder = lambda gp, Z, k, L: bh_dif_cs_sp(Z, gp, k, L)
        result = quad(bh_dif_cs_sp_reorder, 1.0, k-1., args=(Z, k, L))[0]

    return result

def bh_tot_cs_dp(Z, k, T, ni, Zstar):
    '''
    Bethe-Heitler total cross-section assuming screening with a double-exponential potential (Fermi + Debye)
    Integration of Eq.(25) from B. Martinez et al, Physics of Plasmas, 36, 103109 (2019)
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        T : temperature of plasma (keV)
        ni : density of plasma (/m3)
        Zstar : ionization degree of plasma
    outputs :
        result : total cross-section (m^2)
    '''

    condition = (k >= 2.)

    result = 0.0
    if condition:
        # we switch the order of variables for quad
        bh_dif_cs_dp_reorder = lambda gp, Z, k, T, ni, Zstar: bh_dif_cs_dp(Z, gp, k, T, ni, Zstar)
        result = quad(bh_dif_cs_dp_reorder, 1.0, k-1., args=(Z, k, T, ni, Zstar))[0]

    return result