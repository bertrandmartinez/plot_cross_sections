import numpy as np
from scipy.special import zeta
from scipy.constants import c, e, m_e, epsilon_0, mu_0, N_A, k, h, hbar, alpha, value
from screening import g_func, Fermi_length, Debye_length, interatom_length, I1_func, I2_func, reduced_potential_length, Coulomb_correction
from scipy.integrate import quad

# Classical electron radius
r_e = value('classical electron radius')

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
    assuming a potential with a single exponential
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
        b1 = np.sqrt(1.-(1. / g1 ** 2))

        dplus = p1 + p2  # maximum momentum
        dminus = p1 - p2  # minimum momentum

        T1 = 16. * (Z * r_e) ** 2 * alpha / (3. * k * b1 ** 2)

        eta = 1.0 # always true for a single exponential potential
        T2 = g_func(dplus, dminus, eta, L)

        T3 = 1.0
        if elwert :
            T3 = factor_Elwert(Z, k, g1)            

        result = T1 * T2 * T3
    return result

def nr_dif_cs_dp(Z, k, g1, T, ni, Zstar):
    '''
    Computes the non relativistic Bremsstrahlung differential cross-section
    assuming a potential with a sum of two exponentials
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        g1 : Lorentz factor of the incident electron
        L : scale-length of the potential
        elwert : multiply by the Elwert correction (default=True)
    outputs :
        result (m^2) : non relativistic Bremsstrahlung differential cross-section
    '''
    
    condition = (k > 0.) and (k < g1 - 1.) and (g1 > 1.) and (g1 <= 2.)
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

def mr_dif_cs_sp(Z, k, g1, L):
    '''
    Computes the midly relativistic Bremsstrahlung differential cross-section
    assuming a potential with a single exponential
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        g1 : Lorentz factor of the incident electron
        L : scale-length of the potential
        elwert : multiply by the Elwert correction (default=True)
    outputs :
        result (m^2) : midly relativistic Bremsstrahlung differential cross-section
    '''

    condition = (k > 0.) and (k < g1 - 1.) and (g1 >= 2.) and (g1 <= 100.)
    result = 0.0

    if condition:

        d = k / (2. * g1 * (g1-k))  # momentum transfer
        eta = 1.0  # q ratio
        T1 = 4. * (Z * r_e) ** 2 * alpha / k  # factor
        T2 = (1. + ((g1 - k) / g1) ** 2) * (I1_func(d, L, eta) + 1.)
        T3 = (2. / 3.)*(((g1 - k) / g1)) * (I2_func(d, L, eta) + (5. / 6.))
        result = T1 * (T2 - T3)

    return result

def mr_dif_cs_dp(Z, k, g1, T, ni, Zstar):
    '''
    Computes the midly relativistic Bremsstrahlung differential cross-section
    assuming a potential with a sum of two exponentials
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        g1 : Lorentz factor of the incident electron
        L : scale-length of the potential
        elwert : multiply by the Elwert correction (default=True)
    outputs :
        result (m^2) : midly relativistic Bremsstrahlung differential cross-section
    '''

    condition = (k > 0.) and (k < g1 - 1.) and (g1 >= 2.) and (g1 <= 100.)
    result = 0.0

    if condition:

        # Determine the scale-length of the reduced potential
        etaf = 1. - Zstar / Z
        etad = 1. - etaf
        
        Lf = Fermi_length(Z)
        Ld = Debye_length(T, ni, Zstar)  # Debye Length
        Ld = max(interatom_length(ni), Ld)

        # Eq. (22)
        etar = 1.0
        Lr = reduced_potential_length(etaf, Lf, etad, Ld)
        
        d = k / (2. * g1 * (g1-k))  # momentum transfer
        eta = 1.0  # q ratio
        T1 = 4. * (Z * r_e) ** 2 * alpha / k  # factor
        T2 = (1. + ((g1 - k) / g1) ** 2) * (I1_func(d, Lr, etar) + 1.)
        T3 = (2. / 3.)*(((g1 - k) / g1)) * (I2_func(d, Lr, etar) + (5. / 6.))
        result = T1 * (T2 - T3)

    return result

def ur_dif_cs_sp(Z, k, g1, L):
    '''
    Computes the ultra relativistic Bremsstrahlung differential cross-section
    assuming a potential with a single exponential
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        g1 : Lorentz factor of the incident electron
        L : scale-length of the potential
        elwert : multiply by the Elwert correction (default=True)
    outputs :
        result (m^2) : ultra relativistic Bremsstrahlung differential cross-section
    '''

    condition = (k > 0.) and (k < g1 - 1.) and (g1 >= 100.)
    result = 0.0

    if condition:

        fC = Coulomb_correction(Z)
        d = k / (2. * g1 * (g1-k))  # momentum transfer
        eta = 1.0  # q ratio
        T1 = 4. * (Z * r_e) ** 2 * alpha / k  # factor
        T2 = (1. + ((g1 - k) / g1) ** 2) * (I1_func(d, L, eta) + 1. - fC)
        T3 = (2. / 3.)*(((g1 - k) / g1)) * (I2_func(d, L, eta) + (5. / 6.) - fC)
        result = T1 * (T2 - T3)

    return result

def ur_dif_cs_dp(Z, k, g1, T, ni, Zstar):
    '''
    Computes the ultra relativistic Bremsstrahlung differential cross-section
    assuming a potential with a sum of two exponentials
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        g1 : Lorentz factor of the incident electron
        L : scale-length of the potential
        elwert : multiply by the Elwert correction (default=True)
    outputs :
        result (m^2) : ultra relativistic Bremsstrahlung differential cross-section
    '''

    condition = (k > 0.) and (k < g1 - 1.) and (g1 >= 2.) and (g1 <= 100.)
    result = 0.0

    if condition:

        # Determine the scale-length of the reduced potential
        etaf = 1. - Zstar / Z
        etad = 1. - etaf
        
        Lf = Fermi_length(Z)
        Ld = Debye_length(T, ni, Zstar)  # Debye Length
        Ld = max(interatom_length(ni), Ld)

        # Eq. (22)
        etar = 1.0
        Lr = reduced_potential_length(etaf, Lf, etad, Ld)

        fC = Coulomb_correction(Z)
        d = k / (2. * g1 * (g1-k))  # momentum transfer
        eta = 1.0  # q ratio
        T1 = 4. * (Z * r_e) ** 2 * alpha / k  # factor
        T2 = (1. + ((g1 - k) / g1) ** 2) * (I1_func(d, Lr, etar) + 1. - fC)
        T3 = (2. / 3.)*(((g1 - k) / g1)) * (I2_func(d, Lr, etar) + (5. / 6.) - fC)
        result = T1 * (T2 - T3)

    return result

def dif_cs_sp(Z, k, g1, L):
    '''
    Computes the Bremsstrahlung differential cross-section
    assuming a potential with a single exponential
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        g1 : Lorentz factor of the incident electron
        L : scale-length of the potential
        elwert : multiply by the Elwert correction (default=True)
    outputs :
        result (m^2) : Bremsstrahlung differential cross-section
    '''

    result = 0.0
    if (g1 > 1.) and (g1 <= 2.) : # Eq.(8)
        result = nr_dif_cs_sp(Z, k, g1, L)
    elif (g1 > 2.) and (g1 <= 100.) : # Eq.(12)
        result = mr_dif_cs_sp(Z, k, g1, L)
    elif (g1 > 100.) : # Eq.(23)
        result = ur_dif_cs_sp(Z, k, g1, L)

    return result

def dif_cs_dp(Z, k, g1, T, ni, Zstar):
    '''
    Computes the Bremsstrahlung differential cross-section
    assuming a potential with a sum of two exponentials
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        g1 : Lorentz factor of the incident electron
        L : scale-length of the potential
        elwert : multiply by the Elwert correction (default=True)
    outputs :
        result (m^2) : Bremsstrahlung differential cross-section
    '''

    # Choose the formula depending on the energy of the incident electron
    result = 0.0
    
    if (g1 > 1.) and (g1 <= 2.) : # Eq.(8)
        result = nr_dif_cs_dp(Z, k, g1, T, ni, Zstar)
    elif (g1 > 2.) and (g1 <= 100.) : # Eq.(12)
        result = mr_dif_cs_dp(Z, k, g1, T, ni, Zstar)
    elif (g1 > 100.) : # Eq.(23)
        result = ur_dif_cs_dp(Z, k, g1, T, ni, Zstar)

    return result

def cdf_sp_gauss(Z, k, g1, L, if_log):
    '''
    This function computes the CDF of the Bremsstrahlung cross-section
    with Gauss Legendre integration
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        g1 : Lorentz factor of the incident electron
        L : scale-length of the potential
        if_log : boolean to integrate withor without the log variable change
    outputs :
        CDF value
    '''

    if if_log :

        g = lambda x: x * dif_cs_sp(Z, x, g1, L)
        deg = 100

        # Numerator
        a = np.log(1.e-9*(g1-1.))
        b = np.log(k)

        x, w = np.polynomial.legendre.leggauss(deg)
        t = np.exp(0.5 * (x + 1) * (b - a) + a)

        numerator = 0.0
        for j in range(deg):
            numerator += np.sum(w[j] * g(t[j])) * 0.5 * (b - a)

        # Denominator
        a = np.log(1.e-9*(g1-1.))
        b = np.log(g1-1.)

        x, w = np.polynomial.legendre.leggauss(deg)
        t = np.exp(0.5 * (x + 1) * (b - a) + a)

        denominator = 0.0
        for j in range(deg):
            denominator += np.sum(w[j] * g(t[j])) * 0.5 * (b - a)

        result = numerator / denominator

    else :

        f = lambda x: dif_cs_sp(Z, x, g1, L)
        deg = 100

        # Numerator
        a = 0.0
        b = k

        x, w = np.polynomial.legendre.leggauss(deg)
        t = 0.5 * (x + 1) * (b - a) + a

        numerator = 0.0
        for j in range(deg):
            numerator += np.sum(w[j] * f(t[j])) * 0.5 * (b - a)

        # Denominator
        a = 0.0
        b = g1 - 1.0

        x, w = np.polynomial.legendre.leggauss(deg)
        t = 0.5 * (x + 1) * (b - a) + a

        denominator = 0.0
        for j in range(deg):
            denominator += np.sum(w[j] * f(t[j])) * 0.5 * (b - a)

        result = numerator / denominator

    return result

def cdf_sp_quad(Z, k, g1, L):
    '''
    This function computes the CDF of the Bremsstrahlung cross-section
    with quadrature integration
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        g1 : Lorentz factor of the incident electron
        L : scale-length of the potential
        if_log : boolean to integrate withor without the log variable change
    outputs :
        CDF value
    '''

    numerator = quad(lambda k, Z, g1: dif_cs_sp(Z, k, g1, L), 0., k, args=(Z, g1))[0]
    denominator = quad(lambda k, Z, g1: dif_cs_sp(Z, k, g1, L), 0., g1-1., args=(Z, g1))[0]
    result = numerator / denominator
    
    return result

def tot_rcs_quad(Z, g1, L):
    '''
    This function computes the radiative cross-section for Bremstrahlung
    with quadrature integration
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        g1 : Lorentz factor of the incident electron
        L : scale-length of the potential
    outputs :
        CDF value
    '''

    phi = quad(lambda k, Z, g1: k*dif_cs_sp(Z, k, g1, L), 0., g1-1., args=(Z, g1))[0]
    
    return phi

def tot_cs_sp(Z, g1, L):
    '''
    Computes the Bremsstrahlung total cross-section
    assuming a potential with a single exponential
    inputs :
        Z : atomic number
        k : normalized energy of the photon k = hbar \omega / mc^2
        g1 : Lorentz factor of the incident electron
        L : scale-length of the potential
        elwert : multiply by the Elwert correction (default=True)
    outputs :
        result (m^2) : total Bremsstrahlung differential cross-section
    '''

    f = lambda x: x * dif_cs_sp(Z, x, g1, L)
    deg = 100
    
    a = np.log(1.e-9*(g1-1.))
    b = np.log(g1-1.)
    
    x, w = np.polynomial.legendre.leggauss(deg)
    t = np.exp(0.5 * (x + 1) * (b - a) + a)
    
    integral = 0.0
    for j in range(deg):
        integral += np.sum(w[j] * f(t[j])) * 0.5 * (b - a)
    
    return integral