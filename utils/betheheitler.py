from scipy.special import zeta
from scipy.constants import c, e, m_e, epsilon_0, mu_0, N_A, k, h, hbar, alpha, value
from scipy.integrate import quad
from screening import Fermi_length, Debye_length, Coulomb_correction, I1_func, I2_func, reduced_potential_length

# Classical electron radius
r_e = value('classical electron radius')

def dif_cs_sp(Z, gp, k, L):
    '''
    This function computes the differential Bethe-Heitler cross-section
    inputs :
        gp is the energy of the positron
        k is the energy of the photon
        Z is the atomic number
    outputs :
        result is the differential cross-section in m^2
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

def dif_cs_dp(Z, gp, k, L, T, ni, Zstar):
    '''
    This function computes the differential Bethe-Heitler cross-section
    inputs :
        gp is the energy of the positron
        k is the energy of the photon
        Z is the atomic number
    outputs :
        result is the differential cross-section in m^2
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


def bh_cs(Z, k):
    '''
    This function computes the total Bethe-Heitler cross-section
    inputs :
        Z is the atomic number
        k is the energy of the photon
    outputs :
        result is the total cross-section in m^2
    '''

    result = 0.0
    condition = (k > 2.0)

    if condition:
        result = quad(bh_cs_dif, 1.0, k-1.0, args=(k, Z))[0]

    return result


def bh_cdf(Z, k, gp):
    '''
    This function computes the CDF of the Bethe-Heitler cross-section
    inputs :
        Z is the atomic number
        k is the energy of the photon
        gp is the energy of the positron
    outputs :
        result is the CDF of the Bethe-Heitler cs (no units and between 0 and 1 by definition)
    '''

    condition = (k >= 2.) and (gp >= 1.) and (gp <= k-1.)

    result = 0.0
    if condition:
        numerator = quad(bh_cs_dif, 1.0, gp, args=(k, Z))[0]
        denominator = bh_cs(Z, k)
        result = numerator / denominator

    return result