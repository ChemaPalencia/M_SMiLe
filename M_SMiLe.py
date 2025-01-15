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
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from astropy.cosmology import z_at_value
from scipy.interpolate import CubicSpline
from astropy.cosmology import FlatLambdaCDM
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator

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
        assert abs(mu_t) >= mu_r, 'tangential arcs have mu_t >> mu_r!'
        assert mu1 >= 0, 'mu1 must be positive!'
        assert mu_r >= 0, 'mu_r must be positive!'
        assert mu1 < mu2, 'mu1 must be lower than mu2!'
        assert zs > zd, 'Lens plane redshift can not be larger than source plane redshift!'
        assert sigma_star > 0, 'No microlenses. Sigma_star must be greater than 0!'

        # Input parameters
        self.mu_t = mu_t
        self.mu_r = mu_r
        self.sigma_star = sigma_star  # M_sun/pc2
        self.zs = zs
        self.zd = zd
        self.mu1 = mu1
        self.mu2 = mu2

    # Create getter and setters for input parameters ##########################
    @property
    def mu_t(self):
        return self._mu_t

    @mu_t.setter
    def mu_t(self, new_value):
        self._mu_t = new_value

    @property
    def mu_r(self):
        return self._mu_r

    @mu_r.setter
    def mu_r(self, new_value):
        self._mu_r = new_value

    @property
    def sigma_star(self):
        return self._sigma_star

    @sigma_star.setter
    def sigma_star(self, new_value):
        self._sigma_star = new_value

    @property
    def zs(self):
        return self._zs

    @zs.setter
    def zs(self, new_value):
        self._zs = new_value

    @property
    def zd(self):
        return self._zd

    @zd.setter
    def zd(self, new_value):
        self._zd = new_value

    @property
    def mu1(self):
        return self._mu1

    @mu1.setter
    def mu1(self, new_value):
        self._mu1 = new_value

    @property
    def mu2(self):
        return self._mu2

    @mu2.setter
    def mu2(self, new_value):
        self._mu2 = new_value

    @property
    def sigma_eff(self):
        return abs(self._sigma_star * self._mu_t)

    @property
    def mu_m(self):
        return self._mu_r * self._mu_t

    @property
    def sigma_crit(self):
        self.angular_diameter_distances()
        return (0.35*u.g * u.cm**-2
                           * (self.D.to(u.Gpc)/u.Gpc)**-1).to(u.M_sun/u.pc**2).value

    @property
    def sigma_ratio(self):
        return self.sigma_eff / self.sigma_crit
    ###########################################################################

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
        self.D_ds = cosmo.angular_diameter_distance_z1z2(
            z1=self.zd, z2=self.zs)
        self.D = self.D_ds * self.D_d / self.D_s

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
        ln_a = microlenses.lognormal(
            x, A=A, mu=mu_a, sigma=sigma_a, alpha=alpha_a)
        ln_b = microlenses.lognormal(
            x, A=B, mu=mu_b, sigma=sigma_b, alpha=alpha_b)

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
                                          delta, C)*np.heaviside(break1-x, 1)
                                     + right(x, A, break1, break2, exp1, exp2, exp3,
                                             delta, C)*np.heaviside(-break1+x, 0)
                                     / right(break1, A, break1, break2, exp1, exp2, exp3,
                                             delta, C)*left(break1, A, break1, break2, exp1, exp2, exp3,
                                                            delta, C))

        return tot + np.heaviside(x-80, 0)*np.full_like(x, 0)

    @staticmethod
    def powerlaw2BbrokenPowerlawSmooth2(x, A, break1, break2, exp1, exp2, exp3,
                                        delta, **kwargs):
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
        """

        def right(x, A, break1, break2, exp1, exp2, exp3,
                  delta):
            return microlenses.brokenPowerlawSmooth(x, A, break2, exp2, exp3,
                                                    delta)

        def left(x, A, break1, break2, exp1, exp2, exp3,
                 delta):
            return microlenses.powerlaw(x, right(1, A, break1, break2, exp1,
                                                 exp2, exp3, delta), 1, exp1)

        tot = left(x, A, break1, break2, exp1, exp2, exp3, delta) \
            * np.heaviside(break1-x, 1) + right(x, A, break1, break2, exp1,
                                                exp2, exp3, delta) \
            * np.heaviside(-break1+x, 0)

        return tot

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
        pl = A*(x/break_)**(-exp1) \
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

        return np.minimum(np.full_like(x, C), left(x) + center(x) + right(x) +
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
        left = A*(x/break1)**(-exp1) \
            * (0.5*(1+(x/break1)**(1/delta1)))**((exp1-exp2)*delta1)

        exp3 *= -1
        exp4 *= -1
        right = B*(x/break2)**(-exp3) \
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

        # Compute necessary parameters from inputs
        self.Nmu = int(5e4)
        # Mu for computing the whole pdf
        self.log_mu4pdf = np.log10(np.logspace(np.log10(1e-6),
                                               np.log10(5e6),
                                               num=self.Nmu,
                                               endpoint=True, base=10.0,
                                               dtype=np.double, axis=0))
        # Limits where log_mu4pdf ~= log10(mu1) and log10(mu2) values to return
        self.limits_logmu = [np.argmin(abs(10**self.log_mu4pdf-self.mu1*abs(1e3/self.mu_m))),
                             np.argmin(abs(10**self.log_mu4pdf
                                           - self.mu2*abs(1e3/self.mu_m)))]

        self.log_mu4pdf = self.log_mu4pdf[self.limits_logmu[0]: self.limits_logmu[1]]

        # Values unnormalized
        self.log_mu = self.log_mu4pdf + np.log10(abs(self.mu_m)/1000)
        self.mu = 10**self.log_mu

        # Get mass regime for obtaining the correct modeling of the pdf
        self.return_regime()

        # Model and model parameters
        self.models = None
        self.model_params = None

        # Based on the parity and mass regime a modeling is assigned to the object
        x = self.sigma_ratio  # Variable to get the value of the params at.

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

                model1_params = {'A': self.powerlaw(x,
                                                    **pars_val_A)*np.heaviside(-0.05+x, 0)+x**0.8*self.powerlaw(abs(x-0.05),
                                                                                                                **pars_val_A)*np.heaviside(+0.05-x, 1)*np.heaviside(-0.005+x, 0),
                                 'mu_A': self.powerlaw(x,
                                                       **pars_val_mu_A),
                                 'sigma_A': self.powerlaw(x,
                                                          **pars_val_sigma_A),
                                 'B': self.powerlaw(x,
                                                    **pars_val_B),
                                 'mu_B': self.brokenPowerlawSmooth(x,
                                                                   **pars_val_mu_B),
                                 'sigma_B': self.powerlaw(x,
                                                          **pars_val_sigma_B)+2e-2*np.heaviside(0.1-x, 0),
                                 'alpha_B': self.powerlaw(x,
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
                    model2_params['b'] = model2_params['a'] / 591 \
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
                pars_err_sigma_C = {'Cmicrolens.mu_m = mu_t_ * mu_r_': 0.009}

                model3_params = {'C': self.brokenPowerlawSmooth(x,
                                                                **pars_val_C),
                                 'mu_C': self.brokenPowerlawSmooth(x,
                                                                   **pars_val_mu_C),
                                 'sigma_C': self.constant(x,
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

                pars_val_alpha_A = {'A': 0.10170829,
                                    'break1': 1,
                                    'break2': 3.8652296,
                                    'exp1': -0.633, 'exp2': -0.1793595,
                                    'exp3': -11.856396, 'delta': 0.3616786}
                pars_err_alpha_A = {'A': 0.11, 'break1': 0, 'break2': 0.6,
                                    'exp1':  0.00, 'exp2': 0.12, 'exp3':  4,
                                    'delta': 0.08}

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
                                    # 'exp0': -0.214045378,
                                    'exp0': -0.0344,
                                    'exp1': -0.0883842,
                                    'exp2': -0.75870725, 'exp3': -0.13169, 'C': 0}
                pars_err_sigma_B = {'A': 0.003, 'break0': 0.003, 'break1': 0.1,
                                    'break2': 0.2, 'exp0': 0.02, 'exp1': 0.04,
                                    'exp2': 0.13, 'exp3': 0.007, 'C': 0}

                pars_val_alpha_B = {'A': 2.903013647, 'B': 1.1614797669,
                                    'mu': 4.8869619776, 'sigma1': 0.38483396,
                                    'sigma2': 0.42951949799}
                pars_err_alpha_B = {'A': 0.09, 'B': 0.04, 'mu': 0.2,
                                    'sigma1': 0.02, 'sigma2': 0.03}

                model1_params = {'A': self.powerlaw2BbrokenPowerlawSmooth(x,
                                                                          **pars_val_A)*np.heaviside(x-1, 0)
                                 + self.powerlaw(x, 0.92, 1, -0.2578732355108736)*np.heaviside(1-x, 1),
                                 'mu_A': self.threeSmoothPowerlawAndLognormal(x,
                                                                              **pars_val_mu_A)*np.heaviside(x-1, 0)
                                 + self.powerlaw(x, 416, 1, -0.37114263208463394)*np.heaviside(1-x, 1),
                                 'sigma_A': self.fourBrokenPowerlaw(x,
                                                                    **pars_val_sigma_A)*np.heaviside(x-1, 0)
                                 + self.powerlaw(x, 0.3542, 1, -0.030036711843066652)*np.heaviside(1-x, 1),
                                 'alpha_A': self.powerlaw2BbrokenPowerlawSmooth2(x,
                                                                                 **pars_val_alpha_A)*np.heaviside(x-1, 0)
                                 + self.powerlaw(x, 2.34, 1, -0.405799222724421)*np.heaviside(1-x, 1),
                                 'B': self.constminusLognormal2Powerlaw(x,
                                                                        **pars_val_B)*np.heaviside(x-1, 0)
                                 + self.powerlaw(x, 0.28, 1, 4.81763707198723)*np.heaviside(1-x, 1),
                                 'mu_B': self.powerlaw23BbrokenPowerlawSmooth(x,
                                                                              **pars_val_mu_B)*np.heaviside(x-1, 0)
                                 + self.powerlaw(x, 737, 1, -1.641389961757059)*np.heaviside(1-x, 1),
                                 'sigma_B': self.fourBrokenPowerlaw(x,
                                                                    **pars_val_sigma_B)*np.heaviside(x-1, 0)
                                 + self.powerlaw(x, 0.3319, 1, 0.16036329929555113)*np.heaviside(1-x, 1),
                                 'alpha_B': (self.twoConstMinusLognormal(x,
                                                                         **pars_val_alpha_B)*np.heaviside(x-1, 0)*np.heaviside(-x+66.5, 0)
                                             + self.powerlaw(x, 2.31, 1, -0.4005601911082159) *
                                             np.heaviside(1-x, 1)
                                             + self.twoConstMinusLognormal(2*66.5-x, **pars_val_alpha_B)*np.heaviside(
                                     x-66.5, 0))
                                 }
                if x > 125:
                    model1_params['alpha_B'] = 0
                self.model_params = [model1_params]

                if x >= 150:
                    pars_val_B = {'B': 1.2500202, 'break2': 17.5662,
                                  'exp3': 4.904496, 'exp4': 0.02986548,
                                  'delta2': 0.223740118, 'C': 0.741360,
                                  'A': 0.608494, 'break1': 5.717417,
                                  'exp1': 1.817844, 'exp2': 9.9999,
                                  'delta1': 0.127479}
                    model1_params['B'] = self.twoBrokenPowerlawSmooth(x,
                                                                      **pars_val_B)*np.heaviside(x-2.5, 1)

                    pars_val_sigma_B_ = {'A': 0.183939519, 'mu': 4.61129583,
                                         'sigma': 0.2204833636, 'B': 0.14797318,
                                         'b': -0.0629659035}

                    model1_params['sigma_B'] = self.lognormal2powerlaw(x,
                                                                       **pars_val_sigma_B_)
                if (x > 66.5) and (x < 150):
                    pars_val_sigma_B_ = {'A': self.fourBrokenPowerlaw(66.5,
                                                                      **pars_val_sigma_B)*np.heaviside(66.5-1, 0),
                                         'x0': 66.5,
                                         'a': -0.438}

                    model1_params['sigma_B'] = self.powerlaw(x,
                                                             **pars_val_sigma_B_)

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

                # if abs(self.mu_m) < 500:
                model1_params['A_tilde'] = 0

                if model1_params['beta_b'] <= 0.5:
                    model1_params['b'] = 0
                else:
                    model1_params['b'] = model1_params['a'] * (self.sigma_eff
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

                # pars_val_mu_B = {'A': 1084.57652, 'break_': 0.20482544,
                #                  'exp1': 0.0502976, 'exp2': -0.92344,
                #                  'delta': 0.07821}
                pars_val_mu_B = {'A': 1084.57652, 'break_': 0.20482544,
                                 'exp1': 0.0502976, 'exp2': -0.52344,
                                 'delta': 0.07821}
                pars_err_mu_B = {'A': 80, 'break_': 0.03,
                                 'exp1': 0.04, 'exp2': 0.15,
                                 'delta': 0.1}

                pars_val_sigma_B = {'A': 0.999999999, 'x0': 1, 'a': 0.99072515}
                pars_err_sigma_B = {'A': 5e-7, 'x0': 1, 'a': 0.012}

                model2_params = {'B': self.powerlaw(x, **pars_val_B),
                                 'mu_B': self.brokenPowerlawSmooth(x,
                                                                   **pars_val_mu_B),
                                 # 'mu_B': self.powerlaw(x, A=pars_val_mu_B['A'],
                                 #                       exp1=pars_val_mu_B['exp1']),
                                 'sigma_B': self.powerlaw(x,
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

                # pars_val_sigma_C = {'A': 0.332466, 'x0': 1, 'a': 0.2528516} old
                pars_val_sigma_C = {'A': 0.225, 'x0': 1, 'a': 0.2628516}
                pars_err_sigma_C = {'A': 0.011, 'x0': 0, 'a': 0.013}

                model3_params = {'c': 0,
                                 'beta_c': -self.curvature_pl(x,
                                                              **pars_val_beta_c),
                                 'd': self.powerlaw(x, **pars_val_d),
                                 'beta_d': -2,
                                 'C': self.powerlaw(x, **pars_val_C),
                                 'mu_C': self.powerlaw(x, **pars_val_mu_C),
                                 'sigma_C': self.powerlaw(x,
                                                          **pars_val_sigma_C)}

                if model3_params['beta_c'] >= -2:
                    model3_params['c'] = 0
                else:
                    model3_params['c'] = model3_params['d'] * (self.sigma_eff
                                                               / 995)**2

                self.model_params = [model1_params,
                                     model2_params, model3_params]

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

                    if A == 0:
                        return pls(x)

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

                    cut = 10**(np.log10(mu_C) + (2*sigma_C*np.log(10))**2
                               * np.log10(np.exp(1)))

                    return np.heaviside(cut-x, 1)*f1(x) \
                        + np.heaviside(-cut+x, 1)*f2(x)/f2(mu_C)*f1(mu_C)

                self.models = [model1, model2, model3]

            elif self.mass_regime == 'high':
                # Model parameters fit params and stderrs
                pars_val_a = {'C': 0.47538461541080634}
                pars_err_a = {'C': 0.002}

                pars_val_beta_a = {
                    'A': 0.5, 'a': 0.7939284800501769, 'x0': 0.5}
                pars_err_beta_a = {'A': 0, 'a': 0.05, 'x0': 0}

                model1_params = {'a': self.constant(x,
                                                    **pars_val_a)*np.heaviside(50-x, 0),
                                 'beta_a': self.powerlaw(x,
                                                         **pars_val_beta_a)
                                 * np.heaviside(50-x, 0)}

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
                              'delta1': 0.012}

                pars_val_mu_C = {'A': 2780.150065911462, 'break_': 11.00000157,
                                 'exp1': -0.2314221, 'exp2': -0.96796350862}
                pars_err_mu_C = {'A': 100, 'break_':  0.010, 'exp1': 0.02,
                                 'exp2': 0.04}

                pars_val_sigma_C = {'A': 0.183939519, 'mu': 4.61129583,
                                    'sigma': 0.2204833636, 'B': 0.14797318,
                                    'b': -0.0629659035}
                pars_err_sigma_C = {'A': 0.0019, 'mu': 0.05, 'sigma': 0.009,
                                    'B': 0.004, 'b': 0.009}

                model2_params = {'A': (self.threeSmoothPowerlaw(x,
                                                                **pars_val_A)*np.heaviside(x-1, 0)
                                       + self.powerlaw(x, 0.79, 1, 0.2391465751775469)*np.heaviside(1-x, 1))*np.heaviside(15-x, 0),
                                 'mu_A': self.fourBrokenPowerlaw(x,
                                                                 **pars_val_mu_A)*np.heaviside(x-1, 0)
                                 + self.powerlaw(x, 651, 1, -0.10902422577139648)*np.heaviside(1-x, 1),
                                 'sigma_A': self.brokenPowerlawSmooth(x,
                                                                      **pars_val_sigma_A)*np.heaviside(x-1, 0)
                                 + self.powerlaw(x, 0.3692, 1, 0.0) *
                                 np.heaviside(1-x, 1),
                                 'B': (self.lognormals(x,
                                                       **pars_val_B)*np.heaviside(x-1, 0)
                                       + self.powerlaw(x, 0.17, 1, 0.9215198558143352)*np.heaviside(1-x, 1))*np.heaviside(40-x, 0),  # *np.heaviside(55-x, 0),
                                 'mu_B': self.fourBrokenPowerlaw(x,
                                                                 **pars_val_mu_B)*np.heaviside(x-1, 0)
                                 + self.powerlaw(x, 2452, 1, -0.6615682731747226) *
                                 np.heaviside(1-x, 1)+10 *
                                 np.heaviside(x-35, 1) + 5 * \
                                 np.heaviside(x-55, 1) + 5 * \
                                 np.heaviside(x-100, 1),
                                 'sigma_B': self.threeBrokenPowerlawlognormal(x,
                                                                              **pars_val_sigma_B)*np.heaviside(x-1, 0)
                                 + self.powerlaw(x, 0.2728, 1, 0.0) *
                                 np.heaviside(1-x, 1),
                                 'C': self.twoBrokenPowerlawSmooth(x,
                                                                   **pars_val_C)*np.heaviside(x-2.5, 1),
                                 'mu_C': self.twoBrokenPowerlaw(x,
                                                                **pars_val_mu_C),
                                 'sigma_C': self.lognormal2powerlaw(x,
                                                                    **pars_val_sigma_C),
                                 }

                if x >= 125:
                    pars_val_mu_C = {'A': 737.963490, 'break0': 1.000015,
                                     'break1': 2.23622845, 'break2': 4.200702,
                                     'break3': 17.4261488, 'exp0': -1.609319,
                                     'exp1': -0.138034, 'exp2': 1.3166448,
                                     'exp3': -1.397220, 'exp4': -0.18866907,
                                     'delta1': 0.2407108, 'delta2': 0.991359}
                    model2_params['mu_C'] = self.powerlaw23BbrokenPowerlawSmooth(x,
                                                                                 **pars_val_mu_C)*np.heaviside(x-1, 0)

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

                    ln_a = ln_a*np.heaviside(80-self.sigma_ratio, 0)
                    ln_b = ln_b*np.heaviside(80-self.sigma_ratio, 0)

                    return ln_b + ln_a + ln_c

                self.models = [model1, model2]

                if model1_params['a'] == 0:
                    self.models = [model2]
                    self.model_params = [model2_params]

        # Get a curve for each model and find the cuts between them
        self.curves = {}
        for i, model in enumerate(self.models):
            self.curves[f'curve_{i+1}'] = model(10**self.log_mu4pdf,
                                                **self.model_params[i])

        if self.mass_regime == 'low' and self.parity == 1:
            y1 = self.curves['curve_1']
            y2 = self.curves['curve_2']
            y3 = self.curves['curve_3']

            # phuse right and mid first
            argmax = np.log10(self.model_params[-1]['mu_C'])+1
            argmin = np.log10(self.model_params[-1]['mu_C'])-1

            limit_left = np.argmin(abs(self.log_mu4pdf-argmin))
            limit_right = np.argmin(abs(self.log_mu4pdf-argmax))

            y2_ = y2/y3[np.argmax(y3)]
            y3_ = y3/y3[np.argmax(y3)]

            y2_ = y2_[limit_left:limit_right]
            y3_ = y3_[limit_left:limit_right]

            y2_[np.where(y2_ < 1e-8)] = 1e-8
            y3_[np.where(y3_ < 1e-8)] = 1e-8

            min_indices, _ = find_peaks(-np.abs(np.log10(y2_)-np.log10(y3_)))

            transition_index = limit_left + min_indices[-1]
            compound = np.concatenate(
                (y2[:transition_index]/y2[transition_index]*y3[transition_index],
                 y3[transition_index:]))

            # left and remaining
            y2 = compound
            argmax = np.log10(self.model_params[-1]['mu_C'])+.5
            argmin = np.log10(self.model_params[0]['mu_B'])

            limit_left = np.argmin(abs(self.log_mu4pdf-argmin))
            limit_right = np.argmin(abs(self.log_mu4pdf-argmax))

            y1_ = y1/y2[np.argmax(y1)]
            y2_ = y2/y2[np.argmax(y1)]

            y1_ = y1_[limit_left:limit_right]
            y2_ = y2_[limit_left:limit_right]

            y1_[np.where(y1_ < 1e-8)] = 1e-8
            y2_[np.where(y2_ < 1e-8)] = 1e-8

            min_indices, _ = find_peaks(-np.abs(np.log10(y1_)-np.log10(y2_)))

            transition_index = limit_left + min_indices[-1]
            compound = np.concatenate(
                (y1[:transition_index]/y1[transition_index]*y2[transition_index],
                 y2[transition_index:]))

        if self.mass_regime == 'low' and self.parity == -1:
            y1 = self.curves['curve_1']
            y2 = self.curves['curve_2']
            y3 = self.curves['curve_3']

            # phuse right and mid first
            argmax = np.argmax(y2)

            y2_ = y2/y2[argmax]
            y3_ = y3/y2[argmax]

            limit_left = argmax
            limit_right = np.argmin(abs(
                self.log_mu4pdf-np.log10(self.model_params[-1]['mu_C'])))

            x_ = self.log_mu4pdf[limit_left:limit_right]
            y2_ = y2_[limit_left:limit_right]
            y3_ = y3_[limit_left:limit_right]
            y2_[np.where(y2_ < 1e-8)] = 1e-8
            y3_[np.where(y3_ < 1e-8)] = 1e-8

            min_indices, _ = find_peaks(-np.abs(np.log10(y2_)-np.log10(y3_)))
            transition_index = argmax + min_indices[-1]
            compound = np.concatenate(
                (y2[:transition_index]/y2[transition_index]*y3[transition_index],
                 y3[transition_index:]))

            y1 = self.curves['curve_1']
            y2 = compound

            y1_ = y1/y2[argmax]
            y2_ = y2/y2[argmax]

            limit_right = argmax
            limit_left = np.argmin(abs(
                self.log_mu4pdf-np.log10(self.model_params[0]['mu_A'])))

            x_ = self.log_mu4pdf[limit_left:limit_right]
            y1_ = y1_[limit_left:limit_right]
            y2_ = y2_[limit_left:limit_right]
            y1_[np.where(y2_ < 1e-8)] = 1e-8
            y2_[np.where(y1_ < 1e-8)] = 1e-8

            min_indices, _ = find_peaks(-np.abs(np.log10(y2_)-np.log10(y1_)))
            transition_index = limit_left + min_indices[-1]
            compound = np.concatenate(
                (y1[:transition_index], y2[transition_index:]))

        if self.mass_regime == 'high' and self.parity == -1:
            if len(self.curves) > 1:
                y1 = self.curves['curve_1']
                y2 = self.curves['curve_2']

                min_log_diff_index = np.argwhere(y2 >= y1)

                if len(min_log_diff_index) == 0:
                    y1 = y1/1.1
                    min_log_diff_index = np.argwhere(y2 >= y1)

                min_log_diff_index = min_log_diff_index[0][0]

                # Concatenate curves
                compound = np.concatenate(
                    (y1[:min_log_diff_index], y2[min_log_diff_index:]))

                if self.sigma_ratio >= 1:
                    arg_right = np.argmin(abs(
                        np.log10(y2[:np.argmax(y2)]) -
                        np.log10(1.25*y2[min_log_diff_index])))
                    x_right = 10**self.log_mu4pdf[arg_right]
                    y_right = y2[arg_right]

                    def powerlaw_to_fit(x, A, a, pl_params):
                        return self.powerlaw(x, **pl_params) + \
                            self.powerlaw(x, A, x_right, a)

                    A_initial = 1.0
                    a_initial = 1.0

                    pl_params = {'A': self.model_params[0]['a'], 'x0': 10**2.4,
                                 'a': self.model_params[0]['beta_a']}

                    def objective(params):
                        A, a = params
                        return (powerlaw_to_fit(x_right, A, a, pl_params)
                                - y_right)**2

                    bounds = [(y_right/2, y_right),
                              (pl_params['a']*1.5, pl_params['a']*3.5)]
                    result = minimize(
                        objective, [A_initial, a_initial], bounds=bounds)

                    if result.success:
                        y3 = y1 + \
                            self.powerlaw(10**self.log_mu4pdf,
                                          result.x[0], 10**2.4, result.x[1])
                        min_log_diff_index = np.argwhere(y2 >= y3)
                        if len(min_log_diff_index) > 0:
                            min_log_diff_index = min_log_diff_index[0][0]
                            compound = np.concatenate(
                                (y3[:min_log_diff_index], y2[min_log_diff_index:]))

            else:
                compound = self.curves['curve_1']

        if self.mass_regime == 'high' and self.parity == 1:
            compound = self.curves['curve_1']

        # for ii, floor in enumerate([1e-4, 1e-5, 1e-6, 1e-7, 1e-8]):
        #     j = 1
        #     if ii > 2:
        #         j = 2
        #     mask = compound >= floor
        #     pdf_masked = compound[mask]

        #     args_intersec = [np.where(compound >= floor)[0][0],
        #                      np.where(compound >= floor)[0][-1]]

        #     if args_intersec[-1] < len(compound)-1:
        #         x_to_extrapolate1 = self.log_mu[args_intersec[-1] -
        #                                         800*j+1:args_intersec[-1]+1]
        #         y_to_extrapolate1 = np.log10(
        #             compound[args_intersec[-1]-800*j+1:args_intersec[-1]+1])
        #         model1 = CubicSpline(x_to_extrapolate1, y_to_extrapolate1)
        #         x_new1 = self.log_mu[args_intersec[-1]+1:]
        #         y_new1 = 10**model1(x_new1)
        #         if y_new1[-1] > y_new1[-2]:
        #             y_new1[np.argmin(y_new1)-500:] = np.nan
        #     else:
        #         y_new1 = None

        #     if args_intersec[0] > 0:
        #         x_to_extrapolate2 = self.log_mu[args_intersec[0]:
        #                                         args_intersec[0]+800*j]
        #         y_to_extrapolate2 = np.log10(
        #             compound[args_intersec[0]:args_intersec[0]+800*j])
        #         model2 = CubicSpline(x_to_extrapolate2, y_to_extrapolate2)
        #         x_new2 = self.log_mu[:args_intersec[0]]
        #         y_new2 = 10**model2(x_new2)
        #         if y_new2[-1] > y_new2[-2]:
        #             y_new2[:np.argmin(y_new2)+500] = np.nan
        #     else:
        #         y_new2 = None

        #     if y_new1 is None and y_new2 is None:
        #         compound = compound
        #     elif y_new1 is None and y_new2 is not None:
        #         compound = np.concatenate((y_new2, pdf_masked))
        #         del y_new2, x_new2
        #     elif y_new1 is not None and y_new2 is None:
        #         compound = np.concatenate((pdf_masked, y_new1))
        #         del y_new1, x_new1
        #     else:
        #         compound = np.concatenate((y_new2, pdf_masked, y_new1))
        #         del y_new2, y_new1, x_new2, x_new1

        mask = compound >= 1e-7
        compound[~mask] = np.nan
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
        compound, log_mu = self.get_pdf()

        # Set the plot parameters and do the plotting.s
        style_params = {
            'figure.figsize': (8, 8),
            'xtick.major.size': 15,
            'xtick.major.width': 2,
            'xtick.minor.size': 10,
            'xtick.minor.width': 1,
            'xtick.direction': 'in',
            'xtick.top': True,
            'ytick.major.size': 15,
            'ytick.major.width': 2,
            'ytick.minor.size': 10,
            'ytick.minor.width': 1,
            'ytick.direction': 'in',
            'ytick.right': True,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25,
            'xtick.major.pad': 9,
            'legend.fontsize': 30,
            'legend.title_fontsize': 30,
            'axes.titlesize': 30,
            'axes.labelsize': 40,
            'axes.linewidth': 2.5,
            'grid.linewidth': 1,
            'lines.linewidth': 3,
            'lines.solid_capstyle': 'round',
            'font.family': 'serif',
            'text.usetex': True,
            'font.size': 28,
            'font.serif': 'Palatino',
            'legend.frameon': False,
        }
        with plt.rc_context(style_params):
            fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')

            cmap = matplotlib.cm.get_cmap('plasma')

            if self.parity == -1:
                ls = '--'
                color = cmap(0.2)
            elif self.parity == 1:
                ls = '-'
                color = cmap(0.8)

            ax.plot(log_mu, compound, label=f'{self.sigma_ratio:.2f}',
                    lw=3, ls=ls, color=color)

            ax.set_xlim(left=0, right=5)
            ax.set_ylim(bottom=1e-7, top=np.nanmax(compound)*7)

            ax.set_ylabel(r'PDF$\left((\log_{10}\left(\mu\right)\right)$')
            ax.set_xlabel(r'$\log_{10}\left(\mu\right)$')
            ax.set_yscale('log')

            # Saves the image.
            if save_pic:
                if path is None:
                    path = os.getcwd() + '/'
                plt.savefig(path+'magnification_pdf.pdf',
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
                for i, pdf_i in np.ndenumerate(pdf):
                    f.write(str(pdf_i) + ' ' + str(log_mu[i]) + '\n')

        elif extension == 'fits':
            # Write a table with pdf and log mu.
            t = Table()
            t['pdf(log10(mu))'] = pdf
            t['log10(mu)'] = log_mu
            t.write(path+'magnification_pdf.fits', overwrite=True)

        elif extension == 'h5':
            with h5py.File(path+'magnification_pdf.hdf5', 'w') as f:
                dset = f.create_dataset("pdf(log10(mu))", (len(pdf),))
                dset[:] = pdf
                dset = f.create_dataset("log10(mu)", (len(log_mu),))
                dset[:] = log_mu


# If runned by terminal equals True and this part of the code gets to run.
if __name__ == "__main__":

    # Get the values trhough terminal. Help specified.
    parser = argparse.ArgumentParser(
        description='''Given a set of parameters regarding an
                    extragalactic microlensing scheme, this program
                    computes the probability of magnification in a given
                    range.''',
        prog='M-SMiLe.py',
        epilog='Contact: palencia@ifca.unican.es / jpalenciasainz@gmail.com')

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
    assert abs(mu_t) >= 5*mu_r, 'tangential arcs have mu_t >> mu_r!'
    assert sigma_star > 0, 'No microlenses. Sigma_star must be greater than 0!'
    assert mu_r >= 0, 'mu_r must be positive!'

    # An object is instanciated and initialized given the terminal arguments.
    microlens = microlenses(mu_t, mu_r, sigma_star, zs, zd, mu1, mu2)

    # Save data on the .txt file.
    microlens.save_data(path=dir_, extension=extension)

    # Saves and shows the pdf plot.
    if args.plot:
        microlens.plot(save_pic=True, path=dir_)

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
  --extension extension
                        If save, extension in which the data is saved (txt, fits, h5).

Contact: palencia@ifca.unican.es / jpalenciasainz@gmail.com

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
