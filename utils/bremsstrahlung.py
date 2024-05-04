import numpy as np
from scipy.special import zeta
from scipy.constants import c, e, m_e, epsilon_0, mu_0, N_A, k, h, hbar, alpha, value
from screening import g_func, Fermi_length, Debye_length, interatom_length

# Classical electron radius
r_e = value('classical electron radius')

def factor_Elwert(Z, k, g1):
    '''
    Computes the Elwert factor, see Eq.(11)
    '''

    condition = (k > 0.) and (k < g1 - 1.)
    result = 1.0

    if condition :

        g2 = g1 - k
        b1 = np.sqrt(1.-(1. / g1 ** 2))
        b2 = np.sqrt(1.-(1. / g2 ** 2))

        # In theory, the Elwert factor should be applied only if Z*alpha*(1./b2 - 1./b1) << 1
        # In practise, it is closer to Seltzer and Berger data when applied without condition
        #if Z*alpha*(1./b2 - 1./b1) < 0.1 :
        
        result = (b1 / b2) * ((1. - np.exp(-2. * np.pi * Z * alpha / b1)) / (1. - np.exp(-2. * np.pi * Z * alpha / b2)))
    
    return result

def nr_dif_cs_sp(Z, k, g1, L, elwert=True):
    '''
    Computes the nonrelativistic Bremsstrahlung differential cross-section
    assuming a Thomas Fermi's potential
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        g1 : Lorentz factor of the incident electron
        L : scale-length of the potential
        elwert : multiply by the Elwert correction (default=True)
    outputs :
        result (m^2) : nonrelativistic Bremsstrahlung differential cross-section
    '''

    condition = (k > 0.) and (k < g1 - 1.)
    result = 0.0

    if condition:

        g2 = g1 - k
        p1 = np.sqrt(g1 ** 2 - 1.)
        p2 = np.sqrt(g2 ** 2 - 1.)

        dplus = p1 + p2  # maximum momentum
        dminus = p1 - p2  # minimum momentum

        T1 = 16. * (Z * r_e) ** 2 * alpha / (3. * k * p1 ** 2)

        eta = 1.0 # always true for a single exponential potential
        T2 = g_func(dplus, dminus, eta, L)

        T3 = 1.0
        if elwert :
            T3 = factor_Elwert(Z, k, g1)            

        result = T1 * T2 * T3
    return result

def dsig_dk_3BN(Z,g1,k):
    '''
    Formula 3BN from RMP Koch 1959
    inputs:
        Z : atomic number
        g1 : Lorentz factor of the incident electron
        k : normalized energy of the photon k = hbar \omega / mc^2
    outputs :
        result (m^2) : nonrelativistic Bremsstrahlung differential cross-section
    '''

    condition = (k > 0.) and (k < g1 - 1.)
    result = 0.0

    if condition :

        p1 = (g1**2 - 1.)**0.5
        eps0 = np.log((g1+p1)/(g1-p1))
        E = g1 - k
        p = (E**2 - 1.)**0.5 
        eps = np.log((E+p)/(E-p))
        LL = 2.*np.log((g1*E+p1*p-1.)/k)

        term1 = 4./3.-2.*g1*E*((p**2+p1**2)/(p**2*p1**2))+eps0*E/p1**3+eps*g1/p**3-eps*eps0/(p*p1)
        term2 =8.*g1*E/(3.*p1*p)+k**2*(g1**2*E**2+p1**2*p**2)/(p1**3*p**3)
        term3 = k/(2.*p1*p)*(eps0*(g1*E+p1**2)/p1**3-eps*(g1*E+p**2)/p**3+2.*k*g1*E/(p**2*p1**2))
        result = alpha * (Z * r_e)**2*p/p1*(term1+LL*(term2+term3))

    return result

def nr_dif_cs_dp(Z, k, g1, T, ni, Zstar):

    condition = (k > 0.) and (k < g1 - 1.)
    result = 0.0

    if condition:

        g2 = g1 - k
        p1 = np.sqrt(g1 ** 2 - 1.)
        p2 = np.sqrt(g2 ** 2 - 1.)
        b1 = np.sqrt(1.-(1. / g1 ** 2))
        b2 = np.sqrt(1.-(1. / g2 ** 2))

        dplus = p1 + p2  # maximum momentum
        dminus = p1 - p2  # minimum momentum

        etaf = 1.0 - Zstar / Z
        etad = 1.0 - etaf

        Lf = Fermi_length(Z)  # Thomas - Fermi Length
        Ld = Debye_length(T, ni, Zstar)  # Debye Length
        Ld = max(interatom_length(ni), Ld)

        T1 = 16. * (Z * r_e) ** 2 * alpha / (3. * k * p1 ** 2)

        Tf = g_func(dplus, dminus, etaf, Lf)
        Td = g_func(dplus, dminus, etad, Ld)

        G = Lf**2 * np.log(((dminus * Ld) ** 2 + 1.) / ((dplus * Ld) ** 2 + 1.))
        G += Ld**2 * np.log(((dplus * Lf) ** 2 + 1.) / ((dminus * Lf) ** 2 + 1.))
        G *= etaf * etad / (Ld**2 - Lf**2)

        T3 = (b1 / b2) * ((1. - np.exp(-2. * np.pi * Z * alpha / b1)) / (1. - np.exp(-2. * np.pi * Z * alpha / b2)))

        result = T1 * (Tf + Td + G) * T3
    return result