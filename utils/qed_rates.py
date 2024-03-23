import numpy as np
from scipy.special import zeta
from scipy.constants import c, e, m_e, epsilon_0, mu_0, N_A, k, h, hbar, alpha, value

# Classical electron radius
r_e = value('classical electron radius')

def degree(znuc,atwt,rho,te):
  '''
  Fit from More, R., Adv. At. Mol. Phys. 21, 305 (1985). 
  It return the ionization state as a function of 
  znuc : atomic number (Z=29 for copper)
  atwt : atomic weight (63.5 for copper)
  rho : density in g/cm3
  te : temperature in keV
  '''

  alpha=14.3139
  beta=0.6624
  a1=0.003323
  a2=0.971832
  a3=9.26148e-5
  a4=3.10165
  b0=-1.7630
  b1=1.43175
  b2=0.315463
  c1=-0.366667
  c2=0.983333

  t0=1000*te/znuc**(4/3)
  r=rho/(znuc*atwt)
  tf=t0/(1.+t0)
  a=a1*(t0**a2)+a3*(t0**a4)
  b=-np.exp(b0+b1*tf+b2*tf**7)
  c=c1*tf+c2
  q1=a*(r**b)
  q=(r**c+q1**c)**(1/c)
  x=alpha*(q**beta)

  return znuc*x/(1.+x+np.sqrt(1.+2*x))

def ltf(Z):
    '''
    This function determines the Thomas Fermi length for a given atomic number Z
    inputs :
        Z is th atomic number
    outputs :
        result is the TF length, normalised by the compton wavelength
    '''

    compton_wavelength = hbar / (m_e * c)
    length = (4. * np.pi * epsilon_0 * hbar **
              2 / (m_e * e ** 2) * Z ** (-1./3.))
    result = 0.885 * length / compton_wavelength

    return result

def ld(T, ni, Zstar):
    '''
    Debye length normalised by Compton wavelength
    '''

    # T in keV
    # ni is in /m3

    T /= 511.
    compton_wavelength = hbar / (m_e * c)
    length = np.sqrt(T * epsilon_0 * m_e * c**2 / ( e**2 * ni * Zstar * (Zstar + 1.)))

    result = length / compton_wavelength

    return result

def ri(ni):
    '''
    Inter atomic distance, normalised by Compton wavelength
    '''

    # ni in /m3
    compton_wavelength = hbar / (m_e * c)
    result = (3./(4*np.pi*ni))**(1./3.)

    return result / compton_wavelength

def I_1(d, l, q=1.0):
    '''
    This function computes the term I1 in the Bremsstrahlung and Bethe-Heitler cross-section
    inputs :
        d is the delta parameter
        l is the screening length
        q is a fraction between 0 and 1 for this screened potential
    outputs :
        result is the term I1
    '''

    T1 = l * d * (np.arctan(l * d) - np.arctan(l))
    T2 = - (l ** 2 / 2.) * (1. - d) ** 2 / (1. + l ** 2)
    T3 = (1. / 2.) * np.log((1. + l ** 2.) / (1. + (l * d) ** 2))

    result = q ** 2 * (T1 + T2 + T3)  # why q?

    return result

def I_2(d, l, q=1.0):
    '''
    This function computes the term I2 in the Bremsstrahlung and Bethe-Heitler cross-section
    inputs :
        d is the delta parameter
        l is the screening length
        q is a fraction between 0 and 1 for this screenedpotential
    outputs :
        result is the term I2
    '''

    T1 = 4. * (l * d) ** 3 * (np.arctan(d * l) - np.arctan(l))
    T2 = (1. + 3. * (l * d) ** 2) * np.log((1. + l ** 2) / (1. + (l * d) ** 2))
    T3 = (6. * l ** 4 * d ** 2) * np.log(d) / (1. + l ** 2)
    T4 = l ** 2 * (d - 1.) * (d + 1. - 4. * l ** 2 * d ** 2) / (1. + l ** 2)

    result = 0.5 * q * (T1 + T2 + T3 + T4)

    return result


def dsig_dk_3BN(Z,g1,k):
    '''
    Formula 3BN from RMP Koch 1959
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
        result = 7.2973525698E-3 * (Z * 2.8179403267E-15)**2*p/p1*(term1+LL*(term2+term3))

    return result

def dsig_dk_3BNa(Z,g1,k):
    '''
    Formula 3BN(a) from RMP Koch 1959
    '''

    p1 = np.sqrt(g1**2-1)
    b1 = np.sqrt(1.-g1**(-2))

    g2 = g1 - k
    p2 = np.sqrt(g2**2-1)
    b2 = np.sqrt(1.-g2**(-2))

    # Elwert factor
    fEC = (b1 / b2) * ((1. - np.exp(-2. * np.pi * Z * alpha / b1)) / (1. - np.exp(-2. * np.pi * Z * alpha / b2)))  # Elwert correction factor

    result = 16*r_e**2*Z**2*alpha/(3.*b1**2*k)*np.log((p1+p2)/(p1-p2))*fEC
    result = k * result

    return result

def factor_Elwert(Z, k, g1):

    condition = (k > 0.) and (k < g1 - 1.)
    result = 1.0

    if condition :

        g2 = g1 - k
        b1 = np.sqrt(1.-(1. / g1 ** 2))
        b2 = np.sqrt(1.-(1. / g2 ** 2))

        if Z*alpha*(1./b2 - 1./b1) < 100 :
            result = (b1 / b2) * ((1. - np.exp(-2. * np.pi * Z * alpha / b1)) / (1. - np.exp(-2. * np.pi * Z * alpha / b2)))
    
    return result

def nr_cs_dif(Z, k, g1, elwert):
    '''
    This function computes the differential Bremsstrahlung cross-section for the nonrelativistic case
    inputs :
        Z is the atomic number
        k is the energy of the photon
        g1 is the energy of the electron
    outputs :
        differential Bremsstrahlung cross-section for the nonrelativistic case
    '''

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

        L = ltf(Z)  # Thomas - Fermi Length

        T1 = 16. * (Z * r_e) ** 2 * alpha / (3. * k * p1 ** 2)  # factor
        #T1 = 16. * (Z * r_e) ** 2 * alpha / (3. * k * b1 ** 2)  # In Fig. 2 of PoPMartinez2019 : we use 1/b1**2

        T2 = (1. / 2.)*(np.log(((dplus * L) ** 2 + 1.) / ((dminus * L) ** 2 + 1.)) +
                        (1. / ((dplus * L) ** 2 + 1.)) - (1. / ((dminus * L) ** 2 + 1.)))  # glf

        # In Fig. 2 of PoPMartinez2019 : Elwert factor has no condition
        T3 = 1.0
        if elwert :
            T3 = factor_Elwert(Z, k, g1)            

        result = T1 * T2 * T3
    return result


def mr_cs_dif(Z, k, g1):
    '''
    This function computes the differential Bremsstrahlung cross-section for the moderately relativistic case
    inputs :
        Z is the atomic number
        k is the energy of the photon
        g1 is the energy of the electron
    outputs :
        differential Bremsstrahlung cross-section for the moderately relativistic case
    '''

    condition = (k > 0.) and (k < g1 - 1.)

    result = 0.0

    if condition:

        d = k / (2. * g1 * (g1-k))  # momentum transfer

        l = ltf(Z)  # thomas-fermi length

        q = 1.0  # q ratio

        T1 = 4. * (Z * r_e) ** 2 * alpha / k  # factor

        T2 = (1. + ((g1 - k) / g1) ** 2) * (I_1(d, l, q) + 1.)

        T3 = (2. / 3.)*(((g1 - k) / g1)) * (I_2(d, l, q) + (5. / 6.))

        result = T1 * (T2 - T3)

    return result


def ur_cs_dif(Z, k, g1):
    '''
    This function computes the differential Bremsstrahlung cross-section for the ultra relativistic case
    inputs :
        Z is the atomic number
        k is the energy of the photon
        g1 is the energy of the electron
    outputs :
        differential Bremsstrahlung cross-section for the ultra relativistic case
    '''

    condition = (k > 0.) and (k < g1 - 1.)

    result = 0.0

    if condition:

        # initialization of Coulomb correction term
        fc = (alpha * Z) ** 2 / (1. + (alpha * Z) ** 2)

        sum = 0.0

        # loop for zeta function
        for n in range(1, 5):
            sum = sum + ((-alpha * Z) ** 2) ** n * (zeta(2. * n + 1) - 1)

        fc = fc*sum

        d = k / (2. * g1 * (g1 - k))  # momentum transfer

        l = ltf(Z)  # thomas-fermi length

        q = 1.0

        T1 = 4. * (Z * r_e) ** 2 * alpha / k

        T2 = (1.0 + ((g1 - k)/g1) ** 2) * (I_1(d, l, q) + 1.0 - fc)

        T3 = (2. / 3.)*(((g1 - k)/g1)) * (I_2(d, l, q) + (5. / 6.) - fc)

        result = T1 * (T2 - T3)

    return result
