"""
Created on Mon Dec 12 18:16:21 2022

@author: Jose María Palencia

    This program calculates the probability distribution function of the
    magnification factor at microlensing from a galxy cluster on a lensed arcs,
    in a given range of magnification values.
    
    Inputs: mu_t, mu_r, sigma_star, zs, zd
    
        mu_t: Macromodel tangential magnification.
        mu_r: Macromodel radial magnification.
        sigma_star: microlensing surface mass density.
        zs: source plane redshift.
        zd: lens plane redshift.
        
    Optional inputs: mu1, mu2
        mu1: lower bound for the magnification range.
        mu2: upper bound for the magnification range.
            
    Work flow:
        Sigma_crit from zs, zd assuming a flat LCDM cosmology (h=70,Om=30).
        Sigma_eff from mu_m and sigma star.
        Mass regime based on Sigma_eff/Sigma_crit.
        Get the models that described that mass regime.
        Get the parameters of these models based on the mass regime.
        Get the pdf.
        
        Can show plots of the pdf and store them in a txt file in the given range.

    Inputs can be taken from the terminal. Also the class microlenses can be
    imported within a python program where you can call its methods for getting
    the pdfs and plots.
    
"""
# Basic modules
import os
import h5py
import textwrap
import argparse
import warnings
import matplotlib
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import astropy.units as u
from astropy.io import fits
from scipy.special import erf
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy import constants as c
from scipy.signal import savgol_filter
from astropy.cosmology import z_at_value
from astropy.cosmology import FlatLambdaCDM
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator

warnings.filterwarnings("ignore")

class microlenses(object):
    def __init__(self, mu_t, mu_r, sigma_star, zs, zd, mu1=1, mu2=2.5e5):
        """
        Initialization of the object.

        Parameters
        ----------
        mu_t : float
            Macromodel tangential magnification.
        mu_r : float
            Macromodel radial magnification.
        sigma_star : float
            microlensing surface mass density.
        zs : float
            Source plane redshift.
        zd : float
            Lens plane redshift.
        mu1 : float, optional
            Lower bound for the magnification range. The default is 1.
        mu2 : float, optional
            Upper bound for the magnification range. The default is 2.5e5.

        """
        # Make sure the input parameters have a physical meaning
        assert abs(mu_t) >= 10*mu_r, 'tangential arcs have mu_t >> mu_r!'
        assert mu1 >= 0, 'mu1 must be positive!'
        assert mu_r >= 0, 'mu_r must be positive!' 
        assert mu1 < mu2, 'mu1 must be lower than mu2!'
        assert zs > zd, 'Lens plane redshift can not be larger than source plane redshift!'
        assert sigma_star > 0, 'No microlenses. Sigma_star must be greater than 0!'
        
        # Input parameters
        self.mu_t = mu_t
        self.mu_r = mu_r
        self.mu_m = mu_t * mu_r
        self.sigma_star = sigma_star # M_sun/pc2
        self.zs = zs
        self.zd = zd       
        self.mu1 = mu1
        self.mu2 = mu2
        self.Nmu = int(1e5)
        # Mu for computing the whole pdf
        self.log_mu4pdf = np.log10(np.logspace(np.log10(1e-6),
                                               np.log10(5e6),
                                               num=self.Nmu,
                                               endpoint=True, base=10.0,
                                               dtype=np.double, axis=0))
        # Limits where log_mu4pdf ~= log10(mu1) and log10(mu2) values to return
        self.limits_logmu = [np.argmin(abs(10**self.log_mu4pdf-self.mu1*abs(1e3/self.mu_m))),
                             np.argmin(abs(10**self.log_mu4pdf \
                                           -self.mu2*abs(1e3/self.mu_m)))]
        
        self.log_mu4pdf = self.log_mu4pdf[self.limits_logmu[0]: self.limits_logmu[1]]
        
        # Values unnormalized
        self.log_mu = self.log_mu4pdf + np.log10(abs(self.mu_m)/1000)
        self.mu = 10**self.log_mu
        
        # Computed from input parameters
        self.angular_diameter_distances()
        self.critical_surface_density()
        self.sigma_eff()
        
        self.sigma_ratio = self.sigma_eff / self.sigma_crit
        
        # Get mass regime for obtaining the correct modeling of the pdf
        self.return_regime()
        
        # Model and model parameters
        self.models = None
        self.model_params = None
        
    def __str__(self):
        """
        Gives a basic description for the python users :print(object).

        Returns 
        -------
        str
            Object user firendly description.

        """
        text = f'Lens system consisting on microlenses within a galaxy ' \
             + f' cluster at zd = {self.zd}, with a macro-magnification' \
             + f' mu_m = {self.mu_m}, and soruce plane at zs = {self.zs}.' \
             + f' The effective surface mass density is sigma_eff =' \
             + f' {self.sigma_eff:.0f} Msun/pc2, and the critical surface' \
             + f' mass density is sigma_crit = {self.sigma_crit:.0f} Msun/pc2.'
        return textwrap.indent("\n".join(textwrap.wrap(text, 80)), '         ')
    
    def __repr__(self):
        """
        Developer's description allowing to recreate the object.

        Returns
        -------
        str
            Object developer description.

        """
        return f'microlenses(mu_m={self.mu_m}, sigma_star={self.sigma_star},' \
               + f' zs={self.zs}, zd={self.zd}, mu1={self.mu1},' \
               + f' mu2={self.mu2})'

    # def get_mu(self):
    #     """
    #     Returns true mu, self.mu is normalized by (mu -> mu*1000/self.mu_m).

    #     Returns
    #     -------
    #     np.array(dtype=float)
    #         Magnification values [mu1, ..., mu2].

    #     """
    #     return self.mu / abs(1e3/self.mu_m)
    # def get_log_mu(self):
    #     """
    #     Returns true log10(mu), self.log_mu.

    #     Returns
    #     -------
    #     np.array(dtype=float)
    #         Log-magnification values [log10(mu1), ..., log10(mu2)].

    #     """
    #     return self.log_mu - np.log10(abs(1e3/self.mu_m))

    # Compute lens system parameters
    def angular_diameter_distances(self):
        """
        Sets angular diameter distances.

        Sets
        -------
        float
            Angular diameter distances to the lens, source, and between them.

        """
        # Flat LCDM cosmology assumed: H0 = 70, Omega_m = 30
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        
        # Angular diameter distances. Used for getting Sigma_crit
        self.D_d = cosmo.angular_diameter_distance(z=self.zd)
        self.D_s = cosmo.angular_diameter_distance(z=self.zs)
        self.D_ds = cosmo.angular_diameter_distance_z1z2(z1=self.zd,z2=self.zs)
        self.D = self.D_ds * self.D_d / self.D_s
    
    def critical_surface_density(self):
        """
        Sets critical surface mass density.

        Sets
        -------
        float
            Surface mass density for the source-lens plane setting.

        """
        # Assuming a flat LCMD cosmology we get Sigma_crit        
        self.sigma_crit = (0.35*u.g * u.cm**-2 \
                    * (self.D.to(u.Gpc)/u.Gpc)**-1).to(u.M_sun/u.pc**2).value
    
    def sigma_eff(self):
        """
        Sets the effective surface mass density.

        Sets
        -------
        float
            Effective durface mass density. Product of mu_t and sigma_star.

        """
        # Sigma_eff given by the input parameters Sigma_star * |mu_t|
        self.sigma_eff = abs(self.mu_t) * self.sigma_star 

    # Get parity-mass regime
    def return_regime(self):
        """
        Sets the parity and mass regimes from input parameters mu_t and 
        sigma_star.

        Raises
        ------
        Exception
            mu_t can not be 0.

        Sets
        -------
        int
            1 postive parity, -1 negative parity. Depends on the sign of mu_t.
        str
            Surface mass density regime (low, high). Depends on
            Sigma_eff / Sigma_crit.

        """
        # Select parity regime based on the sign of mu_m
        if self.mu_m > 0:
            self.parity = 1
        elif self.mu_m < 0:
            self.parity = -1
        else:
            raise Exception('Introduce a valid macro-model magnification mu_m')
            
        # Select mass regime based on sigma_eff/sigma_crit
        if self.sigma_ratio <= 0.5:
            self.mass_regime = 'low'
        elif self.sigma_ratio > 0.5:
            self.mass_regime = 'high'
    
    # Methods used to represent the scaling of the model parameters w/ Sigma_eff
    @staticmethod
    def powerlaw(x, A, x0, a, **kwargs):
        """
        Powerlaw function

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude.
        x0 : float
            Pivot.
        a : float
            Exponent.
        """
        return A * (x/x0)**a
    
    @staticmethod
    def lognormal(x, A, mu, sigma, alpha, **kwargs):
        """
        Lognomal function

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude.
        mu : float
            Center.
        sigma : float
            Width.
        alpha : float
            Skewness.
        """
        
        x_ = (np.log10(x)-np.log10(mu)) / (np.sqrt(2)*sigma)
        
        return A*np.exp(-(x_**2))*(1+erf(alpha*x_))
    
    @staticmethod
    def lognormals(x, A, mu_a, sigma_a, alpha_a, B, mu_b, sigma_b, alpha_b,
                   **kwargs):
        """
        Lognomal function

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude of lognormal 1.
        mu_a : float
               Center of lognormal 1.
        sigma_a : float
                  Width of lognoraml 1.
        alpha_a : float
                  Skewness oflognormal 1.
        B : float
            Amplitude of lognormal 2.
        mu_b : float
               Center of lognormal 2.
        sigma_b : float
                  Width of lognoraml 2.
        alpha_b : float
                  Skewness oflognormal 2.
        """
        ln_a = microlenses.lognormal(x, A=A, mu=mu_a, sigma=sigma_a, alpha=alpha_a)
        ln_b = microlenses.lognormal(x, A=B, mu=mu_b, sigma=sigma_b, alpha=alpha_b)
        
        return ln_a + ln_b
        
    @staticmethod
    def powerlaw2BbrokenPowerlawSmooth(x, A, break1, break2, exp1, exp2, exp3,
                                       delta, C, **kwargs):
        """
        Powerlaw transition to a smooth broken powerlaw
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude.
        break1 : float
            Transition from powerlaw to smooth broken powerlaw.
        break2 : float
            Break in smooth broken powerlaw.
        exp1 : float
            Exponent powerlaw1.
        exp2 : float
            Exponent 1 smooth broken powerlaw.
        exp3 : float
            Exponent 2 smooth broken powerlaw.
        delta : float
            Smoothness of change parameter.
        C : float
            Constant value at low x
        """        
        def left(x, A, break1, break2, exp1, exp2, exp3,
                                      delta, C):
            return np.minimum(np.full_like(x, C), microlenses.powerlaw(x, A, 1, exp1))
        def right(x, A, break1, break2, exp1, exp2, exp3,
                                      delta, C):
            exp2 *= -1
            exp3 *= -1
            return (x/break2)**(-exp2) \
                 * (0.5*(1+(x/break2)**(1/delta)))**((exp2-exp3)*delta)
                 
        tot = np.heaviside(80-x, 1)*(left(x, A, break1, break2, exp1, exp2, exp3,
                                          delta, C)*np.heaviside(break1-x, 1) \
            + right(x, A, break1, break2, exp1, exp2, exp3,
                    delta, C)*np.heaviside(-break1+x, 0) \
                    /right(break1, A, break1, break2, exp1, exp2, exp3,
                    delta, C)*left(break1, A, break1, break2, exp1, exp2, exp3,
                    delta, C))
                                
        return tot + np.heaviside(x-80, 0)*np.full_like(x, 0)
    
    @staticmethod
    def threeSmoothPowerlawAndLognormal(x, A, exp1, exp2, exp3, break1, break2,
                                        delta1, delta2, B, mu, sigma, alpha, C,
                                        **kwargs):
        """
        Three smooth broken powerlaws plus a lognormal
        
        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Powerlaws amplitude.
        exp1 : float
            Exponent 1 smooth broken powerlaw.
        exp2 : float
            Exponent 2 smooth broken powerlaw.
        exp3 : float
            Exponent 3 smooth broken powerlaw.
        break1 : float
            Break 1 in smooth broken powerlaw.
        break2 : float
            Break 2 in smooth broken powerlaw.
        delta1 : float
            Smoothness of change 1 parameter.
        delta2 : float
            Smoothness of change 2 parameter.
        B : float
            Lognormal amplitude.
        mu : float
            Lognormal center.
        sigma : float
            Lognormal width.
        alpha : float
            Lognormal skewness.
        C : float
            Constant vaue at low x.
        """
        
        exp1 *= -1
        exp2 *= -1
        exp3 *= -1
        
        ln = microlenses.lognormal(x, A, mu, sigma, alpha)
        
        r = A * (x/break1)**(-exp1) \
               * (0.5*(1+(x/break1)**(1/delta1)))**((exp1-exp2)*delta1) \
               * (0.5*(1+(x/break2)**(1/delta2)))**((exp2-exp3)*delta2) + ln
        
        return np.minimum(np.full_like(x, C), r)
        
    @staticmethod
    def complexBrokenPowerlawSmooth(x, A, break1, break2, exp1, exp2, exp3,
                                    delta, **kwargs):
        """
        Powerlaw to smooth broken powerlaw
        
        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude.
        break1 : float
            Transition from powerlaw to smooth broken powerlaw.
        break2 : float
            Break in smooth broken powerlaw.
        exp1 : float
            Exponent powerlaw.
        exp2 : float
            Exponent 1 smooth broken powerlaw.
        exp3 : float
            Exponent 2 smooth broken powerlaw.
        delta : float
            Smoothness of change parameter.
        """
        
        pars = {'A': A, 'break1': break1, 'break2': break2, 'exp1': exp1,
                'exp2': exp2, 'exp3': exp3, 'delta': delta}

        def left(x, A, break1, break2, exp1, exp2, exp3, delta):
            return x**exp1
        def right(x, A, break1, break2, exp1, exp2, exp3, delta):
            exp2 *= -1
            exp3 *= -1
            return A*(x/break2)**(-exp2) \
                 * (0.5*(1+(x/break2)**(1/delta)))**((exp2-exp3)*delta)
                 
        return right(x, **pars)*np.heaviside(-break1+x, 1) \
             + left(x, **pars)*np.heaviside(break1-x, 1)\
             / left(break1, **pars)*right(break1, **pars)
             
    @staticmethod
    def brokenPowerlawSmooth(x, A, break_, exp1, exp2, delta, **kwargs):
        """
        Smooth broken powerlaw

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude.
        break_ : float
            Break in powerlaw smooth.
        exp1 : float
            Exponent 1 smooth broken powerlaw.
        exp2 : float
            Exponent 2 smooth broken powerlaw.
        delta : float
            Smoothness of change parameter.
        """
        exp1 *= -1
        exp2 *= -1

        return A*(x/break_)**(-exp1) \
             * (0.5*(1+(x/break_)**(1/delta)))**((exp1-exp2)*delta)

    @staticmethod
    def twoBrokenPowerlaw(x, A, exp1, exp2, break_, **kwargs):
        """
        Broken powerlaw function

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitde.
        exp1 : float
            Exponent 1.
        exp2 : float
            Exponent 2.
        break_ : float
            Transtion from powerlaw 1 to 2.
        """ 

        def pl1(x):
            return microlenses.powerlaw(x, A, 1, exp1, **kwargs)
        def pl2(x):
            return microlenses.powerlaw(x, pl1(break_), break_, exp2, **kwargs)
        
        def left(x):
            return pl1(x) * np.heaviside(-x+break_, 0)
        
        def right(x):
            return np.heaviside(-break_+x, 0) * pl2(x)
        
        return left(x) + right(x)
                 
    @staticmethod
    def constminusLognormal2Powerlaw(x, A, mu, sigma, exp, break_, **kwargs):
        """
        Function (1 - lognormal) transitioned to a powerlaw

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude.
        mu : float
            Lognormal center.
        sigma : float
            Lognormal width.
        exp : float
            Powerlaw exponent.
        break_ : float
            Transition point.
        """
        def ln(x):
            return microlenses.lognormal(x, A, mu, sigma, 0, **kwargs)
        def pl(x):
            return microlenses.powerlaw(x, 1, 1, exp, **kwargs)
        
        def left(x):
            return (A - ln(x))
        def right(x):
            return pl(x)
        
        return left(x)*np.heaviside(-x+break_, 0) \
              + right(x)*np.heaviside(-break_+x, 1)/right(break_)*left(break_)
              
    @staticmethod
    def powerlaw23BbrokenPowerlawSmooth(x, A, break0, break1, break2, break3, exp0, exp1, exp2,
                                           exp3, exp4, delta1, delta2, **kwargs):
        """
        Broken powerlaw transitiond to a 3 smooth broken powerlaw

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude.
        break0 : float
            Break in broken powerlaw.
        break1 : float
            Transition from powerlaw to smooth broken powerlaws.
        break2 : float
            Break 1 in smooth broken powerlaw.
        break3 : float
            Break 2 in smooth broken powerlaw.
        exp0 : float
            Exponent 1 of broken powerlaw
        exp1 : float
            Exponent 2 of broken powerlaw.
        exp2 : float
            Exponent 1 of smooth broken powerlaw.
        exp3 : float
            Exponent 2 of smooth broken powerlaw.
        exp4 : float
            Exponent 3 of smooth broken powerlaw.
        delta1 : float
            Smoothness of change 1 parameter.
        delta2 : float
            Smoothness of change 2 parameter.
        """
        pars = {'A': A, 'break0': break0, 'break1': break1, 'break2': break2,
                'break3': break3, 'exp0': exp0, 'exp1': exp1, 'exp2': exp2,
                'exp3': exp3, 'exp4': exp4, 'delta1': delta1, 'delta2': delta2}
        
        def left(x, A, break0, break1, break2, break3, exp0, exp1, exp2, exp3,
                 exp4, delta1, delta2):
            return microlenses.powerlaw(x, A, 1, exp0)*np.heaviside(break0-x, 1) + \
                   microlenses.powerlaw(x, microlenses.powerlaw(break0, A, 1, exp0), 1, exp1) \
                 * np.heaviside(x-break0, 0)
        def right(x, A, break0, break1, break2, break3, exp0, exp1, exp2, exp3,
                  exp4, delta1, delta2):
            exp2 *= -1
            exp3 *= -1
            exp4 *= -1
            return (x/break2)**(-exp2) \
                 * (0.5*(1+(x/break2)**(1/delta1)))**((exp2-exp3)*delta1)\
                 * (0.5*(1+(x/break3)**(1/delta2)))**((exp3-exp4)*delta2)
                 
        return left(x, **pars)*np.heaviside(break1-x, 1) \
             + right(x, **pars)*np.heaviside(-break1+x, 0) \
             / right(break1, **pars)*left(break1, **pars)
    @staticmethod
    def threeBrokenPowerlaw(x, A, exp1, exp2, exp3, break1, break2, **kwargs):
        """
        Three broken powerlaw function

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude.
        exp1 : float
            Exponent 1.
        exp2 : float
            Exponent 2.
        exp3 : float
            Exponent 3.
        break1 : float
            Transition from powerlaw 1 to 2.
        break2 : float
            Transition from powerlaw 2 to 3.
        """
        def pl1(x):
            return microlenses.powerlaw(x, A, 1, exp1, **kwargs)
        def pl2(x):
            return microlenses.powerlaw(x, pl1(break1), break1, exp2, **kwargs)
        def pl3(x):
            return microlenses.powerlaw(x, pl2(break2), break2, exp3, **kwargs)
        
        def left(x):
            return pl1(x) * np.heaviside(-x+break1, 0)
        def center(x):
            return np.heaviside(-break1+x, 1)*pl2(x)*np.heaviside(-x+break2, 1)
        def right(x):
            return np.heaviside(-break2+x, 0)*pl3(x)
        
        return left(x) + center(x) + right(x)

    @staticmethod
    def twoConstMinusLognormal(x, A, mu, sigma1, B, sigma2, **kwargs):
        """
        Composite function of two constant minus a lognormal

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Constant 1.
        mu : float
            center of lognormals.
        sigma1 : float
            Lognormal 1 width.
        B : float
            Constant 2.
        sigma2 : float
            Lognormal 2 width.
        """
        def constant_gaussian1(x):
            return A - microlenses.lognormal(x, A, mu, sigma1, 0, **kwargs)
        def constant_gaussian2(x):
            return B - microlenses.lognormal(x, B, mu, sigma2, 0, **kwargs)
        return constant_gaussian1(x) * np.heaviside(mu-x, 0) \
             + constant_gaussian2(x) * np.heaviside(x-mu, 0) \
    
    @staticmethod
    def brokenPowerlawSmooth2constant(x, A, break_, exp1, exp2, delta, C,
                                      **kwargs):
        """
        Smooth broken powerlaw to constant value

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude.
        break_ : float
            Transition from powerlaw 1 to 2.
        exp1 : float
            Exponent 1.
        exp2 : float
            Exponent 2.
        delta : float
            Smoothness of change parameter.
        C : float
            Constant value.
        """
        exp1 *= -1
        exp2 *= -1
        pl =  A*(x/break_)**(-exp1) \
            * (0.5*(1+(x/break_)**(1/delta)))**((exp1-exp2)*delta)
        
        c = np.full_like(x, C)
        
        return np.maximum(pl, c)
    
    @staticmethod
    def fourBrokenPowerlaw(x, A, C, exp1, exp2, exp3, exp0, break1, break2,
                           break0, **kwargs):
        """
        Four borken powerlaws (One added to a constant value)

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude.
        C : float
            Constant value.
        exp1 : float
            Exponent 1.
        exp2 : float
            Exponent 2.
        exp3 : float
            Exponent 3.
        exp0 : float
            Exponent of powerlaw added to constant value.
        break1 : float
            Transition from powerlaw 0 to 1.
        break2 : float
            Transition from powerlaw 1 to 2.
        break0 : float
            Transition from powerlaw 2 to 3.
        """
        
        def pl0(x):
            return microlenses.powerlaw(x, A, 1, exp0, **kwargs) + C
        def pl1(x):
            return microlenses.powerlaw(x, pl0(break0), break0, exp1, **kwargs)
        def pl2(x):
            return microlenses.powerlaw(x, pl1(break1), break1, exp2, **kwargs)
        def pl3(x):
            return microlenses.powerlaw(x, pl2(break2), break2, exp3, **kwargs)
        
        def leftmost(x):
            return pl0(x) * np.heaviside(-x+break0, 1)
        def left(x):
            return np.heaviside(-break0+x, 0) * pl1(x) * np.heaviside(-x+break1, 0)
        def center(x):
            return np.heaviside(-break1+x, 1) * pl2(x) * np.heaviside(-x+break2, 1)
        def right(x):
            return np.heaviside(-break2+x, 0) * pl3(x)
        
        return leftmost(x) + left(x) + center(x) + right(x) 

    @staticmethod    
    def threeBrokenPowerlawlognormal(x, A, exp1, exp2, exp3, break1, break2,
                                     B, mu, sigma, C, **kwargs):
        """
        Lognormal added over a 3 broken powerlaw

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude of powerlaw.
        break1 : float
            Transition from powerlaw 1 to 2.
        break2 : float
            Transition from powerlaw 2 to 3.
        exp1 : float
            Exponent 1.
        exp2 : float
            Exponent 2.
        exp3 : float
            Exponent 3.
        B : float
            Lognormal amplitude.
        mu : float
            Logormal center.
        sigma : float
            Lognormal width.
        C : float
            Constant value at low x
        """
        def powerlaw(x, A, x0, a, **kwargs):
            return A * (x/x0)**a
        
        def pl1(x):
            return powerlaw(x, A, 1, exp1)
        def pl2(x):
            return powerlaw(x, pl1(break1), break1, exp2)
        def pl3(x):
            return powerlaw(x, pl2(break2), break2, exp3)
        
        def left(x):
            return pl1(x) * np.heaviside(-x+break1, 0)
        def center(x):
            return np.heaviside(-break1+x, 1) * pl2(x) * np.heaviside(-x+break2, 1)
        def right(x):
            return np.heaviside(-break2+x, 0) * pl3(x)
        
        return np.minimum(np.full_like(x, C), left(x) + center(x) + right(x) + \
             + B*np.exp(-(np.log10(x)-np.log10(mu))**2/(2*sigma**2)))
                 
    @staticmethod
    def lognormal2powerlaw(x, A, mu, sigma, B, b, **kwargs):
        """
        Lognormal transition to powerlaw

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Lognormal amplitude.
        mu : float
            Lognormal center.
        sigma : float
            Lognormal width.
        B : float
            Powerlaw amplitude.
        b : float
            Powerlaw exponent.
        """
        ln = A*np.exp(-(np.log10(x)-np.log10(mu))**2/(2*sigma**2))
        pl = B * x**b
            
        right = np.maximum(ln, pl)
        
        return ln*np.heaviside(mu-x, 0) + right*np.heaviside(x-mu, 1)
    
    @staticmethod
    def constant(x, C, **kwargs):
        """
        Constant value function

        Parameters
        ----------
        x : float
            Independent value.
        C : float
            Constant value.
        """
        return np.full_like(x, C, dtype='float') * np.heaviside(10-x, 1)
    
    @staticmethod
    def twoBrokenPowerlawSmooth(x, A, break1, exp1, exp2, C, delta1, 
                                B, break2, exp3, exp4, delta2, **kwargs):
        """
        Two smooth broken powerlaw functions, one added to constant value

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude first broken pl.
        break1 : float
            Transition first broken pl.
        exp1 : float
            Exponent 1.
        exp2 : float
            Exponent 2.
        C : float
            Constant value.
        delta1 : float
            Smoothness of change parameter broken pl 1.
        B : float
            Amplitude second broken pl.
        break2 : float
            Transition second broken pl.
        exp3 : float
            Exponent 3.
        exp4 : float
            Exponent 4.
        delta2 : float
            Smoothness of change parameter broken pl 2.
        """
        exp1 *= -1
        exp2 *= -1
        left =  A*(x/break1)**(-exp1) \
              * (0.5*(1+(x/break1)**(1/delta1)))**((exp1-exp2)*delta1)
              
        exp3 *= -1
        exp4 *= -1
        right =  B*(x/break2)**(-exp3) \
              * (0.5*(1+(x/break2)**(1/delta2)))**((exp3-exp4)*delta2) + C
              
        return np.minimum(left, right)
    
    @staticmethod
    def curvature_pl(x, A, x0, alpha, beta, val_cut, **kwargs):
        """
        Powerlaw with curvature function

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude.
        x0 : float
            Pivot value.
        alpha : float
            Base exponent.
        beta : float
            Exponent modifier.
        val_cut : float
            x flor value for the powerlaw
        """
        return np.maximum(A * np.power(x/x0, alpha + beta*(x/x0)), val_cut)
    
    @staticmethod
    def threeSmoothPowerlaw(x, A, exp1, exp2, exp3, break1, break2, delta1,
                          delta2, **kwargs):
        """
        3 smooth broken powerlaws

        Parameters
        ----------
        x : float
            Independent variable.
        A : float
            Amplitude.
        exp1 : float
            Exponent 1.
        exp2 : float
            Exponent 2.
        exp3 : float
            Exponent 2.
        break1 : float
            Transition first broken pl.
        break2 : float
            Transition second broken pl.
        delta1 : float
            Smoothness of change parameter broken pl 1.
        delta2 : float
            Smoothness of change parameter broken pl 2.
        """
        exp1 *= -1
        exp2 *= -1
        exp3 *= -1
        return A*(x/break1)**(-exp1) \
             * (0.5*(1+(x/break1)**(1/delta1)))**((exp1-exp2)*delta1) \
             * (0.5*(1+(x/break2)**(1/delta2)))**((exp2-exp3)*delta2)

    # This method is used to model part of some pdfs
    # (Finds the intersection between two curves)
    @staticmethod
    def find_cut(model1, model2, xmin, xmax, rl, threshold=1e-2, npoints=1e4):
        """
        Finds the value x at which two curves model1(x) and model2(x) intercept.

        Parameters
        ----------
        model1 : function
            Model that creates the first curve.
        model2 : function
            Model that creates the second curve.
        xmin : float
            Lower bound for the range at which we search the intercept.
        xmax : float
            Upper bound for the range at which we search the intercept.
        rl : int
            1 if we search for the right-most intercept, -1 for the left-most.
        threshold : float, optional
            Minimum distance between curves to consider a cut between them.
            The default is 1e-2.
        npoints : float, optional
            Number of points at which we evaluate the curves. Takes the floor
            integer. The default is 1e4.

        Returns
        -------
        return_val : float
            Value x within [xmin, xmax] at which the intercept that we want is
            given.

        """
        # Linear interpolation
        xs = np.linspace(xmin, xmax, num=int(npoints), endpoint=True)
    
        # Model curves
        curve1 = model1(xs)
        curve2 = model2(xs)
    
        # Absolute value of the difference
        res = abs(curve1 - curve2)
    
        max_peak = xs[np.argmax(curve1)]
    
        if rl == 1:
            # Values at right of the peak
            res = res[xs > max_peak]
            return_val = xs[np.argmin(res) + len(xs) - len(res)]
    
        else:
            # Values at left of the peak
            res = res[xs <= max_peak]
            return_val = xs[np.argmin(res)]
    
        return return_val
    
    def get_pdf(self):
        """
        From the mass regime, get the model and parameters that
        represent it, and returns the pdf(log10(mu)).

        Parameters
        ----------

        Returns
        -------
        np.array(pdf)
            pdf(log10(mu)) Probability of magnification.

        """
        # Based on the parity and mass regime a modeling is assigned to the object
        x = self.sigma_ratio # Variable to get the value of the params at.

        # Model params obtained from the scalings based on sigma_ratio
        # Parameter are drawn from dictionaries with name as keys and value 
        # from the corresponding functions
        if self.parity == 1:
            if self.mass_regime == 'low':
                # Model parameters fit params and stderrs
                pars_val_A = {'A': 0.17868334920, 'x0': 1, 'a': -0.5359720999}
                pars_err_A = {'A': 0.019, 'x0': 0, 'a': 0.03}
                
                pars_val_mu_A = {'A': 573.24522072, 'x0': 1, 'a': -0.119289226}
                pars_err_mu_A = {'A': 11, 'x0': 0, 'a': 0.005}
                
                pars_val_sigma_A = {'A': 0.08729187, 'x0': 1, 'a': 0.150218}
                pars_err_sigma_A = {'A': 0.006, 'x0': 0, 'a': 0.018}
                
                pars_val_B = {'A': 0.5411805, 'x0': 1, 'a': -0.86569717}
                pars_err_B = {'A': 0.03, 'x0': 0, 'a': 0.02}
                
                pars_val_mu_B = {'A': 667.7720147, 'break_': 0.164338583,
                                 'exp1': -5.59757091e-05, 'exp2': -0.593964471,
                                 'delta': 1.006911}
                pars_err_mu_B = {'A': 200, 'break_': 0.17, 'exp1': 0.007,
                                 'exp2': 0.27, 'delta': 0.366}
                
                pars_val_sigma_B = {'A': 0.747113776, 'x0': 1, 'a': 0.92611}
                pars_err_sigma_B = {'A': 0.06, 'x0': 0, 'a': 0.03}
                
                pars_val_alpha_B = {'A': 5.5380, 'x0': 1, 'a': 0.47461554}
                pars_err_alpha_B = {'A': 0.602, 'x0': 0, 'a': 0.07}
                
                model1_params = {'A' : self.powerlaw(x,
                                            **pars_val_A)*np.heaviside(0.1-x, 0),
                                 'mu_A' : self.powerlaw(x,
                                               **pars_val_mu_A),                                 
                                 'sigma_A' : self.powerlaw(x,
                                                  **pars_val_sigma_A),
                                 'B' : self.powerlaw(x,
                                            **pars_val_B),
                                 'mu_B' : self.brokenPowerlawSmooth(x,
                                               **pars_val_mu_B),
                                 'sigma_B' : self.powerlaw(x,
                                                  **pars_val_sigma_B),
                                 'alpha_B' : self.powerlaw(x,
                                                  **pars_val_alpha_B)}
                
                pars_val_a = {'A': 0.3051608026, 'x0': 1, 'a': 0.8398767}
                pars_err_a = {'A': 0.04, 'x0': 0, 'a': 0.05}
                
                pars_val_beta_b = {'A': 6.9543945046, 'break_': 0.0808229394,
                                   'exp1': -2.1344044576210308e-07,
                                   'exp2': -0.5854889, 'delta': 0.689493}
                pars_err_beta_b = {'A': 1.1, 'break_': 0.04,
                                   'exp1': 0.1e-07, 'exp2': 0.17,
                                   'delta': 0.3}
                
                model2_params = {'a': self.powerlaw(x,
                                            **pars_val_a),
                                 'beta_b': -self.brokenPowerlawSmooth(x,
                                               **pars_val_beta_b),
                                 'beta_a': -2,
                                 'b': 0}
                    
                if model2_params['beta_b'] >= -2:
                    model2_params['b'] = 0
                else:
                    model2_params['b'] =  model2_params['a'] / 591 \
                                        * np.sqrt(self.sigma_eff)
                      
                pars_val_C = {'A': 0.0842881609, 'break_': 0.114330,
                              'exp1': 1.52000, 'exp2': 0.322239, 'delta': 0.25}
                pars_err_C = {'A': 0.03, 'break_': 0.04, 'exp1': 0.17, 
                              'exp2': 0.3, 'delta': 0.0006}    
                
                pars_val_mu_C = {'A': 1388.925865, 'break_': 0.05033655,
                                 'exp1': -0.3974666, 'exp2': 0, 'delta': 0.01}
                pars_err_mu_C = {'A': 60, 'break_': 0.006, 'exp1': 0.09, 
                                 'exp2': 0, 'delta': 0.005}  
                
                pars_val_sigma_C = {'C': 0.3527344840835316}
                pars_err_sigma_C = {'C': 0.009}  
            
                                     
                model3_params = {'C' : self.brokenPowerlawSmooth(x,
                                            **pars_val_C),
                                 'mu_C' : self.brokenPowerlawSmooth(x,
                                               **pars_val_mu_C),                                 
                                 'sigma_C' : self.constant(x,
                                                  **pars_val_sigma_C)}
                
                
                # Model functions
                def model1(x, A, mu_A, sigma_A, B, mu_B, sigma_B, alpha_B,
                           **kwargs):
                    ln_a = microlenses.lognormal(x, A, mu_A, sigma_A, 0)
                    ln_b = microlenses.lognormal(x, B, mu_B, sigma_B, alpha_B)
                
                    return ln_a + ln_b

                def model2(x, a, beta_a, b, beta_b, **kwargs): 
                    pl_a = microlenses.powerlaw(x, a, 10**3.5, beta_a)
                    pl_b = microlenses.powerlaw(x, b, 10**3.5, beta_b)
                
                    return pl_a + pl_b
                
                def model3(x, C, mu_C, sigma_C, **kwargs): 
                    ln_c = microlenses.lognormal(x, C, mu_C, sigma_C, 0)
                
                    return ln_c

                self.models = [model1, model2, model3]
                self.model_params = [model1_params, model2_params,
                                     model3_params]

            elif self.mass_regime == 'high':
                # Model parameters fit params and stderrs
                pars_val_A = {'A': 0.9195577, 'break1': 2.284070657,
                              'break2': 2.80529,
                              'exp1': -0.97489335, 'exp2': 4.07280959,
                              'exp3': -0.5967299499, 'delta': 0.23093,
                              'C': 1.0066192}
                pars_err_A = {'A': 0.02, 'break1': 0.04, 'break2': 0.07,
                              'exp1':  0.07, 'exp2': 0.3, 'exp3':  0.02,
                              'delta': 0.019, 'C': 0.014}
                
                pars_val_mu_A = {'A': 67.1068, 'exp1': -0.3515, 'exp2': 0.58267,
                                 'exp3': -1.418526, 'break1': 5.1235559,
                                 'break2': 34.185965, 'delta1': 0.04472928,
                                 'delta2': 0.9577237, 'B': 156.602696,
                                 'mu': 3.26330575, 'sigma': 0.2817684,
                                 'alpha': 1.8096824, 'C': 475.9999}
                pars_err_mu_A = {'A':  5, 'exp1':  0.07, 'exp2': 0.4,
                                 'exp3':  1.1, 'break1': 6, 'break2': 30,
                                 'delta1': 0.3, 'delta2': 0.05, 'B': 14,
                                 'mu': 0.08, 'sigma': 0.016, 'alpha': 0.5,
                                 'C': 13}
            
                pars_val_sigma_A = {'A': 0.3545063, 'break0': 1.0573248,
                                    'break1': 19.24305, 'break2': 49.608957,
                                    'exp0': -0.068901, 'exp1': -0.4286963,
                                    'exp2': -0.5820749, 'exp3': 0, 'C': 0}
                pars_err_sigma_A = {'A': 0.003, 'break0': 0.04, 'break1': 6,
                                    'break2': 6, 'exp0': 0.02, 'exp1': 0.007,
                                    'exp2': 0.08, 'exp3': 0, 'C': 0}
                
                pars_val_alpha_A = {'A': 0.10170829, 'break_': 3.8652296,
                                    'exp1': -0.1793595, 'exp2': -11.856396,
                                    'delta': 0.3616786}
                pars_err_alpha_A = {'A': 0.11, 'break_': 0.6, 'exp1':  0.12,
                                    'exp2':  4, 'delta': 0.08}
                
                pars_val_B = {'A': 1.1999999, 'mu': 0.5765915, 'sigma': 0.32205671,
                              'exp': 0.33, 'break_': 5.4}
                pars_err_B = {'A': 0.0000005, 'mu': 0.014, 'sigma': 0.008,
                              'exp': 0.013, 'break_': 0.3}
                
                pars_val_mu_B = {'A': 737.963490, 'break0': 1.000015, 'break1': 2.23622845, 'break2': 4.200702,
                                 'break3': 17.4261488, 'exp0': -1.609319, 'exp1': -0.138034, 'exp2': 1.3166448,
                                 'exp3': -1.397220, 'exp4': -0.18866907,
                                 'delta1': 0.2407108, 'delta2': 0.991359}
                pars_err_mu_B = {'A': 10, 'break0': 0.014, 'break1': 0.05,
                                 'break2': 0.3, 'break3': 7, 'exp0': 0.02,
                                 'exp1': 0.04, 'exp2': 0.3, 'exp3': 0.3, 
                                 'exp4': 0.008, 'delta1': 0.05, 'delta2': 0.06}
                
                pars_val_sigma_B = {'A': 0.33199, 'break0': 1.2302728,
                                    'break1': 2.24730336, 'break2': 3.3912697,
                                    'exp0': -0.214045378, 'exp1': -0.0883842,
                                    'exp2': -0.75870725, 'exp3': -0.13169, 'C': 0}
                pars_err_sigma_B = {'A': 0.003, 'break0': 0.003, 'break1': 0.1,
                                    'break2': 0.2, 'exp0': 0.02, 'exp1': 0.04,
                                    'exp2': 0.13, 'exp3': 0.007, 'C': 0}
                
                pars_val_alpha_B = {'A': 2.903013647, 'B': 1.1614797669,
                                    'mu': 4.8869619776, 'sigma1': 0.38483396,
                                    'sigma2': 0.42951949799}
                pars_err_alpha_B = {'A': 0.09, 'B': 0.04, 'mu': 0.2,
                                    'sigma1': 0.02, 'sigma2': 0.03}
                
                model1_params = {'A' : self.powerlaw2BbrokenPowerlawSmooth(x,
                                            **pars_val_A),
                                 'mu_A' : self.threeSmoothPowerlawAndLognormal(x,
                                               **pars_val_mu_A),                                 
                                 'sigma_A' : self.fourBrokenPowerlaw(x,
                                                  **pars_val_sigma_A),
                                 'alpha_A' : self.brokenPowerlawSmooth(x,
                                                  **pars_val_alpha_A),
                                 'B' : self.constminusLognormal2Powerlaw(x,
                                            **pars_val_B),
                                 'mu_B' : self.powerlaw23BbrokenPowerlawSmooth(x,
                                               **pars_val_mu_B),
                                 'sigma_B' : self.fourBrokenPowerlaw(x,
                                                  **pars_val_sigma_B),
                                 'alpha_B' : self.twoConstMinusLognormal(x,
                                                  **pars_val_alpha_B)}
                self.model_params = [model1_params]

                # Model functions
                def model1(x, A, mu_A, sigma_A, alpha_A, 
                              B, mu_B, sigma_B, alpha_B, **kwargs):
                
                    ln_a = microlenses.lognormal(x, A, mu_A, sigma_A, alpha_A)
                    ln_b = microlenses.lognormal(x, B, mu_B, sigma_B, alpha_B)
                
                    return ln_a + ln_b
                
                self.models = [model1]

        elif self.parity == -1:
            if self.mass_regime == 'low':
                # Model parameters fit params and stderrs
                pars_val_A_tilde = {'A': 137.289030, 'break_': 0.167092667,
                                    'exp1': 0.8620705, 'exp2': 1.23573214e-08,
                                    'delta': 0.1}
                pars_err_A_tilde = {'A': 9, 'break_': 0.03, 'exp1': 0.2,
                                    'exp2': 6e-05, 'delta': 0.00012}
                
                pars_val_mu_A_tilde = {'A': 0.7379049009694179, 'x0': 1,
                                       'a': 2.1772660412562974}
                pars_err_mu_A_tilde = {'A': 0.10, 'x0': 0, 'a': 0.10}
                
                pars_val_sigma_A4 = {'A': 0.1773112, 'delta': 0.60708,
                                     'break_': 0.063271881,
                                    'exp1': 0.19562840, 'exp2': 0.08096189}
                pars_err_sigma_A4 = {'A': 0.05, 'break_': 0.03,
                                     'exp1': 0.15, 'exp2': 0.12, 'delta': 2}
            
                pars_val_alpha_A = {'A': 4.20547529, 'break_': 0.2163274854,
                                    'exp1': 0.0734200, 'exp2': -2.1830373,
                                    'delta': 0.3427642}
                pars_err_alpha_A = {'A': 1.3, 'break_': 0.06, 'exp1': 0.04,
                                    'exp2': 0.8, 'delta': 0.13}
                
                pars_val_a = {'A': 0.957067828, 'x0': 1, 'a': 0.768268354}
                pars_err_a = {'A': 0.07, 'x0': 0, 'a': 0.03}
                
                pars_val_beta_b = {'A': 4.271952475, 'x0': 0.0929109, 
                                   'alpha': -0.2486057, 'beta': -0.393434852,
                                   'val_cut': 0.5}
                pars_err_beta_b = {'A': 3, 'x0': 0.10, 
                                   'alpha': 0.17, 'beta': 0.6,
                                   'val_cut': 0}
                
                model1_params = {'A_tilde': self.brokenPowerlawSmooth(x, 
                                                           **pars_val_A_tilde),
                                 'mu_A_tilde': self.powerlaw(self.mu_r,
                                                       **pars_val_mu_A_tilde), 
                                 'sigma_A4': self.brokenPowerlawSmooth(x,
                                                **pars_val_sigma_A4),
                                 'A': None,
                                 'mu_A': None, 
                                 'sigma_A': None,
                                 'alpha_A': self.brokenPowerlawSmooth(x,
                                                           **pars_val_alpha_A),
                                 'a': self.powerlaw(x, **pars_val_a),
                                 'beta_a': 0.5,
                                 'b': 0,
                                 'beta_b': self.curvature_pl(x,
                                                            **pars_val_beta_b)}

                if model1_params['beta_b'] <= 0.5:
                    model1_params['b'] = 0
                else:
                    model1_params['b'] =  model1_params['a'] * (self.sigma_eff \
                                        / 1928)**1.7678
                
                def R_sigma4(mu_r):
                    return -0.57999*np.tanh((mu_r-2.8)*2.674)+1.5790
                
                def amp_true(amp):
                    return amp / abs(self.mu_m)
                
                def mu_true(mu):
                    return mu * 1000 / abs(self.mu_m)

                model1_params['A'] = amp_true(model1_params['A_tilde'])   
                model1_params['mu_A'] = mu_true(model1_params['mu_A_tilde'])   
                model1_params['sigma_A'] = R_sigma4(self.mu_r) \
                                               * model1_params['sigma_A4']
                
                pars_val_B = {'A': 0.2533466981, 'x0': 1, 'a': -1.019007532129}
                pars_err_B = {'A': 0.03, 'x0': 0, 'a': 0.04}
                
                pars_val_mu_B = {'A': 1084.57652, 'break_': 0.20482544, 
                                 'exp1': 0.0502976, 'exp2': -0.92344,
                                 'delta': 0.07821}
                pars_err_mu_B = {'A': 80, 'break_': 0.03, 
                                 'exp1': 0.04, 'exp2': 0.15,
                                 'delta': 0.1}
                
                pars_val_sigma_B = {'A': 0.999999999, 'x0': 1, 'a': 0.99072515}
                pars_err_sigma_B = {'A': 5e-7, 'x0': 1, 'a': 0.012}                
                
                model2_params = {'B' : self.powerlaw(x, **pars_val_B),
                                 'mu_B' : self.brokenPowerlawSmooth(x,
                                                              **pars_val_mu_B),
                                 'sigma_B' : self.powerlaw(x,
                                                           **pars_val_sigma_B)}
                
                pars_val_d = {'A': 0.491805512267, 'x0': 1, 'a': 0.9768118836}
                pars_err_d = {'A': 0.05, 'x0': 0, 'a': 0.04}
                
                pars_val_beta_c = {'A': 3.691505023, 'x0': 0.1474638, 
                                   'alpha': -0.3331486, 'beta': -0.91630,
                                   'val_cut': 2}
                pars_err_beta_c = {'A': 0.9, 'x0': 0.03, 
                                   'alpha': 0.06, 'beta': 0.3,
                                   'val_cut': 0}
                
                pars_val_C = {'A': 0.08957077, 'x0': 1, 'a': 1.3879966}
                pars_err_C = {'A': 0.004, 'x0': 0, 'a': 0.019}
                
                pars_val_mu_C = {'A': 5336.83523, 'x0': 1, 'a': -0.16549351}
                pars_err_mu_C = {'A': 400, 'x0': 0, 'a': 0.03}
                
                pars_val_sigma_C = {'A': 0.332466, 'x0': 1, 'a': 0.2528516}
                pars_err_sigma_C = {'A': 0.011, 'x0': 0, 'a': 0.013}
                
                model3_params = {'c' : 0,
                                 'beta_c' : -self.curvature_pl(x, 
                                                            **pars_val_beta_c),
                                 'd' : self.powerlaw(x, **pars_val_d),
                                 'beta_d': -2,
                                 'C' : self.powerlaw(x, **pars_val_C),
                                 'mu_C' : self.powerlaw(x, **pars_val_mu_C),
                                 'sigma_C' : self.powerlaw(x,
                                                           **pars_val_sigma_C)}
                
                if model3_params['beta_c'] >= -2:
                    model3_params['c'] = 0
                else:
                    model3_params['c'] = model3_params['d'] * (self.sigma_eff \
                                       / 995)**2
                    
                self.model_params = [model1_params, model2_params, model3_params]
                                               
                # Model functions
                def model1(x, A, mu_A, sigma_A, alpha_A, a, beta_a, b, beta_b,
                           **kwargs):

                    def ln(x, A=A, mu_A=mu_A, sigma_A=sigma_A, alpha_A=alpha_A):
                        return microlenses.lognormal(x, A, mu_A, sigma_A,
                                                     alpha_A)
                        
                    def pls(x, a=a, beta_a=beta_a, b=b, beta_b=beta_b):
                        pl1 = microlenses.powerlaw(x, a, 10**2.4, beta_a)
                        pl2 = microlenses.powerlaw(x, b, 10**2.4, beta_b)
                        return pl1 + pl2

                    max_ = x[np.argmax(ln(x))]
                    epsilon = self.find_cut(ln, pls, max_-0.5, max_+0.5, -1)

                    def right(x):
                        return (ln(x) + pls(x)) * np.heaviside(x-epsilon, 1)
                
                    def left(x):
                        A = (ln(epsilon) + pls(epsilon)) \
                          / ln(epsilon) * np.heaviside(-x+epsilon, 0)
                        return A * ln(x)
                
                    return right(x) + left(x)                            
                
                def model2(x, B, mu_B, sigma_B):
                    return microlenses.lognormal(x, B, mu_B, sigma_B, 0)
                
                def model3(x, c, beta_c, d, beta_d, C, mu_C, sigma_C, **kwargs):
                   
                    def f1(x):
                        pl1 = microlenses.powerlaw(x, c, 10**3.5, beta_c)
                        pl2 = microlenses.powerlaw(x, d, 10**3.5, beta_d)
                        ln = microlenses.lognormal(x, C, mu_C, sigma_C, 0)
                        return pl1 + pl2 + ln
                
                    def f2(x):
                        ln = microlenses.lognormal(x, C, mu_C, sigma_C, 0)
                        return ln
                
                    cut = 10**(np.log10(mu_C) + (2*sigma_C*np.log(10))**2 \
                               * np.log10(np.exp(1)))
                
                    return np.heaviside(cut-x, 1)*f1(x) \
                         + np.heaviside(-cut+x, 1)*f2(x)/f2(mu_C)*f1(mu_C)
                            
                self.models = [model1, model2, model3]

            elif self.mass_regime == 'high':
                # Model parameters fit params and stderrs
                pars_val_a = {'C':0.47538461541080634}
                pars_err_a = {'C':0.002}
                
                pars_val_beta_a = {'A': 0.5, 'a': 0.7939284800501769, 'x0': 0.5}
                pars_err_beta_a = {'A': 0, 'a': 0.05, 'x0': 0}
                
                model1_params = {'a': self.constant(x,
                                           **pars_val_a)*np.heaviside(20-x, 0),
                                 'beta_a': self.powerlaw(x,
                                                **pars_val_beta_a)\
                                           *np.heaviside(20-x, 0)}
                                
                pars_val_A = {'A': 0.4850714, 'break1': 1.71558,
                              'break2': 7.94933, 'exp1': 0.46677,
                              'exp2': -1.191556862, 'exp3': -7.8194305,
                              'delta1': 0.3361, 'delta2': 0.1002}
                pars_err_A = {'A': 0.2, 'break1': 0.19, 'break2': 1.5, 
                              'exp1': 0.18, 'exp2': 0.2, 'exp3': 7,
                              'delta1': 0.14, 'delta2': 0.006}
                
                pars_val_mu_A = {'A': 641.4182, 'C': 0, 
                                 'exp0': 0.275689, 'exp1': -0.45483,
                                 'exp2': -0.960455, 'exp3': 0,
                                 'break0': 1.02003, 'break1': 2.82823,
                                 'break2': 5.57538}
                pars_err_mu_A = {'A': 7, 'C': 0, 
                                 'exp0': 0.03, 'exp1': 0.03,
                                 'exp2': 0.07, 'exp3': 0,
                                 'break0': 0.03, 'break1': 0.16,
                                 'break2': 0.2}
                
                pars_val_sigma_A = {'A': 0.1799199462, 'break_': 4.473668659,
                                    'exp1': -0.4324212, 'exp2': -1.24964013,
                                    'delta': 0.1}
                pars_err_sigma_A = {'A': 0.011, 'break_': 0.4, 'exp1': 0.011,
                                    'exp2': 0.18, 'delta': 0.00001}
                
                pars_val_B = {'A': 1.089066683, 'mu_a': 4.509535,
                              'sigma_a': 0.3403977, 'B': 1.10552985398,
                              'mu_b': 15.578558479, 'sigma_b': 0.136937276,
                              'alpha_a': 0, 'alpha_b': 0}
                pars_err_B = {'A': 0.019, 'mu1': 0.16, 'sigma1': 0.015,
                              'B': 0.06, 'mu2': 0.19, 'sigma2': 0.007,
                              'alpha_a': 0, 'alpha_b': 0}
                
                pars_val_mu_B = {'A': 1685.418437999, 'C': 719.0570099, 
                                 'exp0': -1.63378369, 'exp1': -1.35958269,
                                 'exp2': 0.072643258, 'exp3': -1.624556358,
                                 'break0': 4.65659617, 'break1': 6.556202959,
                                 'break2': 20.147220259}
                pars_err_mu_B = {'A': 50, 'C': 30, 'exp0': 0.03,
                                 'exp1': 0.4, 'exp2': 0.1, 'exp3': 0.8,
                                 'break0': 0.2, 'break1': 0.6, 'break2': 3}
                
                pars_val_sigma_B = {'A': 0.2720893, 'B': 0.109044389,
                                    'C': 0.3515, 'exp1': -1.1437071,
                                    'exp2': 0.05095869, 'exp3': -0.5379241,
                                    'break1': 0.99999, 'break2': 2.739624695,
                                    'mu': 8.494897629, 'sigma': 0.148531189}
                pars_err_sigma_B = {'A': 0.007, 'B': 0.007, 'C': 0.006, 
                                    'exp1': 0.3, 'exp2': 0.04, 'exp3': 0.03,
                                    'break1': 0.012, 'break2': 0.05,
                                    'mu': 0.17, 'sigma': 0.012}
                
                pars_val_C = {'B': 1.2500202, 'break2': 17.5662,
                              'exp3': 4.904496, 'exp4': 0.02986548,
                              'delta2': 0.223740118, 'C': 0.741360, 
                              'A': 0.608494, 'break1': 5.717417,
                              'exp1': 1.817844, 'exp2': 9.9999,
                              'delta1': 0.127479}
                pars_err_C = {'B': 0.6, 'break2': 3, 'exp3': 2, 'exp4': 0.08,
                              'delta2': 0.07, 'C': 0.06, 'A': 0.07,
                              'break1': 0.7, 'exp1': 1.6, 'exp2': 0.018,
                              'delta1':0.012 }
                
                pars_val_mu_C = {'A': 2780.150065911462, 'break_': 11.00000157,
                                 'exp1': -0.2314221, 'exp2': -0.96796350862}
                pars_err_mu_C = {'A': 100, 'break_':  0.010, 'exp1': 0.02,
                                 'exp2': 0.04}
                
                pars_val_sigma_C = {'A': 0.183939519, 'mu': 4.61129583,
                                    'sigma': 0.2204833636, 'B': 0.14797318,
                                    'b': -0.0629659035}
                pars_err_sigma_C = {'A': 0.0019, 'mu': 0.05, 'sigma': 0.009,
                                    'B': 0.004, 'b': 0.009}
                
                model2_params = {'A' : self.threeSmoothPowerlaw(x,
                                            **pars_val_A)*np.heaviside(15-x,0),
                                 'mu_A' : self.fourBrokenPowerlaw(x,
                                               **pars_val_mu_A),
                                 'sigma_A' : self.brokenPowerlawSmooth(x,
                                                  **pars_val_sigma_A),
                                 'B' : self.lognormals(x,
                                            **pars_val_B),
                                 'mu_B' : self.fourBrokenPowerlaw(x,
                                                 **pars_val_mu_B),
                                 'sigma_B' : self.threeBrokenPowerlawlognormal(x,
                                                  **pars_val_sigma_B),
                                 'C' : self.twoBrokenPowerlawSmooth(x,
                                            **pars_val_C)*np.heaviside(x-2.5,1),
                                 'mu_C' : self.twoBrokenPowerlaw(x,
                                                            **pars_val_mu_C),
                                 'sigma_C' : self.lognormal2powerlaw(x,
                                                  **pars_val_sigma_C),
                                 }
                
                self.model_params = [model1_params, model2_params]
                
                # Model functions
                def model1(x, a, beta_a, **kwargs):
                    return microlenses.powerlaw(x, a, 10**2.4, beta_a)
                
                def model2(x, A, mu_A, sigma_A, B, mu_B, sigma_B, C, mu_C, 
                           sigma_C, **kwargs):
                
                    ln_a = microlenses.lognormal(x, A, mu_A, sigma_A, 0)
                    ln_b = microlenses.lognormal(x, B, mu_B, sigma_B, 0)
                    ln_c = microlenses.lognormal(x, C, mu_C, sigma_C, 0)
                    ln_a[np.argwhere(np.isnan(ln_a))] = 0
                    ln_b[np.argwhere(np.isnan(ln_b))] = 0
                    ln_c[np.argwhere(np.isnan(ln_c))] = 0
                
                    return ln_a + ln_b + ln_c
                
                self.models = [model1, model2]
                
                if model1_params['a'] == 0:
                    self.models = [model2]
                    self.model_params = [model2_params]
                    
        
        # Get a curve for each model and find the cuts between them
        curves = {}
        for i, model in enumerate(self.models):
            curves[f'curve_{i+1}'] = model(10**self.log_mu4pdf,
                                           **self.model_params[i])

        ncuts = len(self.models) -1
                
        if ncuts == 0:
            compound = curves['curve_1']
        else:
            
            limits = np.zeros(ncuts)
            arg_mu = np.arange(len(self.log_mu4pdf))

            for i, ele in enumerate(self.models):
                if i < len(self.models)-1:
                    curve_1 = curves[f'curve_{i+1}']
                    curve_2 = curves[f'curve_{i+2}']
                    dif = abs(curve_1-curve_2)/np.minimum(curve_1, curve_2)
                    arg_min = np.argmin(dif)
                    
                    if self.parity == 1 and self.mass_regime == 'low':
                        if i == 0:
                            if self.sigma_ratio >= 0.09:
                                extra = 250
                            else:
                                extra = 0
                            arg_peak = np.argmax(curve_1)
                            dif = np.abs(np.log10(curve_1[np.argmax(curve_1)+extra:]) \
                                      -np.log10(curve_2[np.argmax(curve_1)+extra:]))
                            arg_min = np.argmin(dif) + arg_peak
                            
                            limits[i] = arg_min
                        
                    if self.parity == -1:
                        if ncuts == 1: # High sigma_eff
                            limits[i] = arg_min
                        else: # Low sigma_eff
                            arg_max = np.nanargmax(curves['curve_2'])
                            if i == 0:
                                if self.sigma_ratio >= 0.09:
                                    extra = 500
                                else:
                                    extra = 0
                                dif = np.abs(np.log10(curve_1[:arg_max-extra])\
                                             -np.log10(curve_2[:arg_max-extra]))
                                arg_min = np.nanargmin(dif)
                                if self.log_mu4pdf[arg_min] <= np.log10(self.model_params[0]['mu_A']):
                                    dif = np.abs(np.log10(curve_1[arg_min+100:arg_max-extra])\
                                             -np.log10(curve_2[arg_min+100:arg_max-extra]))
                                        
                                    arg_min = np.nanargmin(dif) + arg_min+100
                            else:
                                dif = np.abs(np.log10(curve_1[arg_max+extra//2:])\
                                             -np.log10(curve_2[arg_max+extra//2:]))
                                arg_min = np.nanargmin(dif) + arg_max

                            limits[i] = arg_min
                                
                    else: # Postive and low sigma_eff
                        limits[i] = arg_min
            
            limits = limits.astype('int')
            
            compound = np.ones(len(self.log_mu4pdf))
            
            # Set the curves together into one single array assert continuity
            if ncuts > 0:
                if ncuts == 1:
                    # high neg
                    compound[0:limits[0]+1] = curves['curve_1'][0:limits[0]+1] \
                  / curves['curve_1'][limits[0]] * curves['curve_2'][limits[0]]

                    compound[limits[0]:] = curves['curve_2'][limits[0]:]
                else:
                    for i in range(ncuts+1):
                        if i == 0:
                            compound[0:limits[0]+1] = curves['curve_1'][0:limits[0]+1]
                        elif i == 1:
                            compound[limits[0]:limits[1]+1] = compound[limits[0]] \
                            / curves['curve_2'][limits[0]] \
                            * curves['curve_2'][limits[0]:limits[1]+1]
                        else:
                            compound[limits[1]:] = compound[limits[1]] \
                            / curves['curve_3'][limits[1]] \
                            * curves['curve_3'][limits[1]:]
            else:
                compound = curves['curve_1']
                        
        return compound, self.log_mu
    
    # Plots the pdf
    def plot(self, save_pic=False, path=None):
        """
        Plots and save the image of the pdf(log_10(mu)) vs mu.

        Parameters
        ----------
        save_pic : bool, optional
            If true the image is saved in the given path. The default is False.
        path : str, optional
            Path to save the image. The default is the cwd.

        Plots (saves)
        -------
        Probability of magnification in the given range.

        """
       
        # Get the pdf based on the mass regime.
        compound = self.get_pdf()[0]
        
        # Masks negligible values (probability < 10**(-5)).
        masked_pdf = ma.masked_where(compound<1e-5, compound, copy=True)
        masked_mu = ma.masked_where(compound<1e-5, self.log_mu, copy=True)
        masked_mu = 10**masked_mu
        
        # Set the plot parameters and do the plotting.s
        with plt.rc_context({"xtick.major.pad": 4}):
            fig, ax = plt.subplots(figsize=(10,6))
            
            cmap = matplotlib.cm.get_cmap('plasma')

            if self.parity == -1:
                ls = '--'
                color = cmap(0.2)
            elif self.parity == 1:
                ls = '-'
                color = cmap(0.8)
            
            ax.plot(masked_mu, masked_pdf, label=r'pdf$(\log_{10}(\mu))$',
                    lw=3, ls=ls, color=color)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(r'$\mu$', fontsize=18)
            ax.set_ylabel(r'pdf$(\log_{10}(\mu))$', fontsize=18)
            ax.tick_params(which='major', length=4, labelsize=12,
                                direction="in")
            ax.tick_params(which='minor', length=2, labelsize=12,
                direction="in")
            
            y_minor = mpl.ticker.LogLocator(base=10.0,
                                subs=np.arange(1.0, 10.0)*0.1, numticks=10)
            ax.yaxis.set_minor_locator(y_minor)
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            
            x_minor = mpl.ticker.LogLocator(base=10.0,
                                subs=np.arange(1.0, 10.0)*0.1, numticks=10)
            ax.xaxis.set_minor_locator(x_minor)
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            
            plt.tight_layout()
            
            # Saves the image.
            if save_pic:
                if path is None:
                    path = os.getcwd() + '/'
                    plt.savefig(f'magnification_pdf.pdf',
                                dpi=300)
            
            # Displays the image.
            plt.show()        
            
            
            
    def save_data(self, path=None, extension='txt'):
        """
        Save the values of pdf(log_10(mu)) and log10(mu) in a txt file at the
        given path.

        Parameters
        ----------
        path : str, optional
            Path to save the .txt file. The default is the cwd.
        extension : str, optional
            Extension to save the data. The default is a txt file.

        Saves
        -------
        file with the pdf and log10(mu) values in the given range.

        """
        
        # Set the path
        if path is None:
                path = os.getcwd() + '/'
        
   
        pdf, log_mu = self.get_pdf()
        
        if extension == 'txt':
            # Write a row pdf and log mu space-separated per mu.
            with open(path+'magnification_pdf.txt', 'w') as f:
                f.write('pdf(log10(mu)) log10(mu)\n')
                for i, pdf in np.ndenumerate(pdf):
                    f.write(str(pdf) + ' ' + str(self.log_mu[i]) + '\n')
        
        elif extension == 'fits':
            # Write a table with pdf and log mu.
            t = Table()
            t['(log10(mu))'] = pdf
            t['log10(mu)'] = self.log_mu
            t.write(path+'magnification_pdf.fits', overwrite=True)
            
        elif extension == 'h5':
            with h5py.File(path+'magnification_pdf.hdf5', 'w') as f:
                dset = f.create_dataset("pdf(log10(mu))", (len(pdf),))
                dset[:] = pdf
                dset = f.create_dataset("log10(mu)", (len(self.log_mu),))
                dset[:] = self.log_mu

# If runned by terminal equals True and this part of the code gets to run.
if __name__ == "__main__":
    
    # Get the values trhough terminal. Help specified.
    parser = argparse.ArgumentParser(
                    description='''Given a set of parameters regarding an
                    extragalactic microlensing scheme, this program
                    computes the probability of magnification in a given
                    range.''',
                    prog='M-SMiLe.py',
                    epilog = 'Contact: palencia@ifca.unican.es')
    
    # Mandatory parameters    
    parser.add_argument('mu_t', metavar='mu_t', type=float,
                        help='Value of the tangential macro-magnification.',
                        action='store')
    parser.add_argument('mu_r', metavar='mu_r', type=float,
                        help='Value of the radial macro-magnification.',
                        action='store')
    
    parser.add_argument('sigma_star', metavar='sigma_star',
                        type=float, action='store',
                        help='Surface mass density of microlenses [Msun/pc2].')
    
    parser.add_argument('zd', metavar='zd', type=float,
                        help='Redshift at the lens plane (cluster).',
                        action='store')
    parser.add_argument('zs', metavar='zs', type=float,
                        help='Redshift at the source plane.',
                        action='store')
    
    # Optional parameters
    parser.add_argument('--mu1', metavar='mu1', type=float,
                        help='Minimum magnification to display the pdf.',
                        action='store', default=1, required=False)
    parser.add_argument('--mu2', metavar='mu2', type=float,
                        help='Maximum magnification to display the pdf.',
                        action='store', default=2.5e5, required=False)
    
    parser.add_argument('--dir', nargs='?', default=os.getcwd()+'/',
                        help='Directory where the results will be stored.',
                        required=False)
    
    parser.add_argument('--plot', metavar='plot', type=bool,
                        help='If "True", plot and save the pdf.',
                        action='store', default=True, required=False)
    
    parser.add_argument('--save', metavar='save', type=bool,
                        help='If "True", save the pdf in a file.',
                        action='store', default=True, required=False)
    
    parser.add_argument('--extension', metavar='extension', type=str,
                        help='If save, extension in which the data is saved (txt, fits, h5).',
                        action='store', default='txt', required=False)

    # Retrive the parameters given through the terminal.
    args = parser.parse_args()
    
    mu_t = args.mu_t
    mu_r = args.mu_r
    sigma_star = args.sigma_star
    zs = args.zs
    zd = args.zd
    mu1 = args.mu1
    mu2 = args.mu2
    dir_ = args.dir
    save_pic = args.save
    extension = args.extension
    
    # Make assertions on the arguments, otherwise an exception is raised.
    assert mu1 >= 0, 'mu1 must be positive!'    
    assert mu1 < mu2, 'mu1 must be lower than mu2!'
    assert zs > zd, 'zd redshift can not be larger than zs!'
    assert abs(mu_t) >= 10*mu_r, 'tangential arcs have mu_t >> mu_r!'
    assert sigma_star > 0, 'No microlenses. Sigma_star must be greater than 0!'
    assert mu_r >= 0, 'mu_r must be positive!' 

    # An object is instanciated and initialized given the terminal arguments.
    microlens = microlenses(mu_t, mu_r, sigma_star, zs, zd, mu1, mu2)
 
    # Save data on the .txt file.
    microlens.save_data(path=dir_, extension=extension)
    
    # Saves and shows the pdf plot.
    if args.plot:
        microlens.plot()
    
# Help provided for terminal access.    
"""

usage: M-SMiLe.py [-h] [--mu1 mu1] [--mu2 mu2] [--dir [DIR]] [--plot plot] [--save save] mu_t mu_r sigma_star zd zs

Given a set of parameters regarding an extragalactic microlensing scheme, this program computes the probability of magnification in a given
range.

positional arguments:
  mu_t         Value of the tangential macro-magnification.
  mu_r         Value of the radial macro-magnification.
  sigma_star   Surface mass density of microlenses [Msun/pc2].
  zd           Redshift at the lens plane (cluster).
  zs           Redshift at the source plane.

optional arguments:
  -h, --help            show this help message and exit
  --mu1 mu1             Minimum magnification to display the pdf.
  --mu2 mu2             Maximum magnification to display the pdf.
  --dir [DIR]           Directory where the results will be stored.
  --plot plot           If "True", plot and save the pdf.
  --save save           If "True", save the pdf in a file.
  --extension extension If save, extension in which the data is saved (txt, fits, h5).

Contact: palencia@ifca.unican.es

"""

# Terminal call example
"""

python3 M_SMiLe.py -600 2 5 1 1.7 --dir /foo/bar/test/ --save False --mu2 1000

"""

# Python call example
"""

from M_SMiLe import microlenses

microlens = microlenses(mu_t=-200, mu_r=5, sigma_star=12, zs=1.3, 
                        zd=0.7, mu1=1e-3, mu2=1e5)

microlens.plot(save_pic=True)

pdf, log_mu = microlens.get_pdf()

"""