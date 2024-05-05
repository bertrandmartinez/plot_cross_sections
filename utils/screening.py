import numpy as np
from scipy.special import zeta, lambertw
from scipy.constants import c, e, m_e, epsilon_0, mu_0, N_A, k, h, hbar, alpha, value

# Classical electron radius
r_e = value('classical electron radius')

def fit_More_zstar(znuc,atwt,rho,te):
    '''
    Fit from More, R., Adv. At. Mol. Phys. 21, 305 (1985). 
    It return the ionization state of a plasma
    inputs : 
        znuc : atomic number (Z=29 for copper)
        atwt : atomic weight (63.5 for copper)
        rho : density in g/cm3
        te : temperature in keV
    outputs :
        result : the ionization state of the plasma
    '''

    alpha = 14.3139
    beta = 0.6624
    a1 = 0.003323
    a2 = 0.971832
    a3 = 9.26148e-5
    a4 = 3.10165
    b0 = -1.7630
    b1 = 1.43175
    b2 = 0.315463
    c1 = -0.366667
    c2 = 0.983333

    t0 = 1000. * te / znuc**(4. / 3.)
    r = rho / (znuc * atwt)
    tf = t0 / (1. + t0)
    a = a1 * t0**a2 + a3 * t0**a4
    b = -np.exp(b0 + b1 * tf + b2 * tf**7)
    c = c1 * tf + c2
    q1 = a * r**b
    q = (r**c + q1**c)**(1. / c)
    x = alpha * q**beta

    result = znuc * x / (1. + x + np.sqrt(1. + 2. * x))

    return result

def g_func(dplus, dminus, eta, L):
    '''
    This function returns the intermediate function g in Eq.(9)
    inputs :
        dplus : maximum momentum transfer
        dminus : minimum momentum transfer
        eta : coefficient in (0,1)
        L : scale-length of the potential
    outputs :
        result : g function
    '''

    T1 = np.log(((dplus * L) ** 2 + 1.) / ((dminus * L) ** 2 + 1.))
    T2 = (1. / ((dplus * L) ** 2 + 1.))
    T3 = (1. / ((dminus * L) ** 2 + 1.))

    result = (eta**2 / 2.) * (T1 + T2 - T3)

    return result

def Fermi_length(Z):
    '''
    Thomas Fermi length, see Eq.(2)
    inputs :
        Z is th atomic number
    outputs :
        result : Thomas Fermi length, normalised by the compton wavelength
    '''

    compton_wavelength = hbar / (m_e * c)
    length = (4. * np.pi * epsilon_0 * hbar **
              2 / (m_e * e ** 2) * Z ** (-1./3.))
    result = 0.885 * length / compton_wavelength

    return result

def Debye_length(T, ni, Zstar):
    '''
    Debye length, see Eq.(4)
    inputs :
        T : plasma temperature in keV
        ni : plasma density in m^{-3}
        Zstar : ionization degree of the plasma
    outputs :
        result : Debye length, normalised by the compton wavelength
    '''

    # T in keV
    # ni is in /m3

    T /= 511.
    compton_wavelength = hbar / (m_e * c)
    length = np.sqrt(T * epsilon_0 * m_e * c**2 / ( e**2 * ni * Zstar * (Zstar + 1)))
    
    result = length / compton_wavelength
    
    result = max(interatom_length(ni), result)

    return result

def interatom_length(ni):
    '''
    Interatomic distance
    inputs :
        ni : ion density in m^{-3}
    outputs :
        result : Interatomic distance normalized by Compton wavelength
    '''
    # ni in /m3
    compton_wavelength = hbar / (m_e * c)
    length = (3./(4*np.pi*ni))**(1./3.)
    result = length / compton_wavelength
    
    return result

def I1_func(d, L, eta):
    '''
    This function computes the term I1 in the Bremsstrahlung and Bethe-Heitler cross-section
    inputs :
        d is the delta parameter
        l is the screening length
        q is a fraction between 0 and 1 for this screened potential
    outputs :
        result is the term I1
    '''

    T1 = L * d * (np.arctan(L * d) - np.arctan(L))
    T2 = - (L ** 2 / 2.) * (1. - d) ** 2 / (1. + L ** 2)
    T3 = (1. / 2.) * np.log((1. + L ** 2.) / (1. + (L * d) ** 2))

    result = eta ** 2 * (T1 + T2 + T3)

    return result

def I2_func(d, L, eta):
    '''
    This function computes the term I2 in the Bremsstrahlung and Bethe-Heitler cross-section
    inputs :
        d is the delta parameter
        l is the screening length
        q is a fraction between 0 and 1 for this screenedpotential
    outputs :
        result is the term I2
    '''

    T1 = 4. * (L * d) ** 3 * (np.arctan(d * L) - np.arctan(L))
    T2 = (1. + 3. * (L * d) ** 2) * np.log((1. + L ** 2) / (1. + (L * d) ** 2))
    T3 = (6. * L ** 4 * d ** 2) * np.log(d) / (1. + L ** 2)
    T4 = L ** 2 * (d - 1.) * (d + 1. - 4. * L ** 2 * d ** 2) / (1. + L ** 2)

    result = 0.5 * eta * (T1 + T2 + T3 + T4)

    return result

def reduced_potential_length(etaf, Lf, etad, Ld):
    '''
    Returns a length Lr from Eq.(22)
    inputs :
        etaf : etaf = 1 - Zstar / Z
        Lf : scale-length of Fermi potential
        etad : etaf = Zstar / Z
        Ld : scale-length of Debye potential
    output:
        result : the length Lr from Eq.(22)
    '''

    # ITFD from Eq.(19)
    T1 = (etaf**2 / 2.) * ( (1. + Lf**2) * np.log(1. + Lf**2) - Lf**2) / (1. + Lf**2)
    T2 = (etad**2 / 2.) * ( (1. + Ld**2) * np.log(1. + Ld**2) - Lf**2) / (1. + Ld**2)
    T3 = etaf * etad * ( Ld**2 * np.log(1. + Lf**2) - Lf**2 * np.log(1. + Ld**2)) / (Ld**2 - Lf**2)
    ITFD = T1 + T2 + T3

    # notation
    a = ITFD
    
    # Eq.(22)
    Lr = np.sqrt(np.exp(lambertw(-np.exp(-(1. + 2. * a))) + 1. + 2.*a) - 1.)
    Lr = np.real(Lr)
    #print(Lr)
    
    #compton_wavelength = hbar / (m_e * c)
    #Lr /= compton_wavelength
    
    return Lr

def Coulomb_correction(Z):
    '''
    Returns the Coulomb correction Eq.(24)
    inputs :
        Z : atomic number
    outputs :
        fC : Coulomb correction
    '''

    fC = 0.0
    for n in range(1, 5):
        fC += ((-alpha * Z) ** 2) ** n * (zeta(2. * n + 1) - 1)
        
    fC *= (alpha * Z) ** 2 / (1. + (alpha * Z) ** 2)

    return fC