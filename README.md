# M_SMiLe
This repository contains code that computes an approximation of the probability of magnification for a lens system consisting of microlensing by compact objects within a galaxy cluster.
The code specifically focuses on the scenario where the galaxy cluster is strongly lensing a background galaxy, and the compact objects, such as stars, are sensitive to this microlensing effect.
The microlenses responsible for this effect are stars and stellar remnants, but also exotic objects such as compact dark matter candidates (PBHs, axion mini-halos...) can contribute to this effect.

More information about this code can be obtained from the paper: URL

# Input parameters
This code generates the magnification probability for a system with the desired input parameters, these are:

$z_{\rm s}$: Redshift of the source plane.

$z_{\rm d}$: Redshift of the lens plane. These two combined give, assuming an standard $\Lambda\rm{CDM}$ cosmology, the critical surface mass density, $\Sigma_{\rm crit}$, of the system.

$\mu_{\rm r}$: Radial macro-magnification of the strongly lensed images of the source.

$\mu_{\rm t}$: Tangential macro-magnification of the strongly lensed images of the source. Can be either positive or negative.

$\Sigma_{\ast}$: Surface mass density of microlenses. The product with $\left|\mu_{\rm t}\right|$ gives the effective surface mass density, $\Sigma_{\rm eff}$, that determines the model used to compute the magnification probability.

$\mu_1$: Lower limit to compute the mgnification probability.

$\mu_2$: Upper limit to compute the magnification probability.

# Outputs
The magnification probability values at different magnification bins are saved to a file of the desired extension, as a two-column .txt file, as a fits table, or as an hdf5 group with two data sets.

In addition, users have the option to generate a plot of the magnification probability curves saved as a .pdf file.

# Installation
To use this code you need Python. This code has been written and tested with Python 3.9 but older versions should work.

To install and use this code, follow the steps below:
1. Starting the terminal
2. Clone the repository:
```
$ https://github.com/ChemaPalencia/M_SMiLe.git
```
3. Change into the project directory:
```
$ cd M_SMiLe
```
4. Install the required dependencies. It is recommended to set up a virtual environment before installing the dependencies to avoid conflicts with other Python packages:
```
$ pip install -r requirements.txt
```
or
```
$ !conda install --file requierements.txt
```

# Usage
This code can be used in two independent ways:

* Via terminal.
  
This code can work as a black box that takes the necessary inputs and generated different files with the desire input.

A detailed description of all the parameters and its effects can be obatined trough:
```
$ python M_SMiLe.py -h
usage: M-SMiLe.py [-h] [--mu1 mu1] [--mu2 mu2] [--dir [DIR]] [--plot plot]
              [--save save] [--extension extension]
              mu_t mu_r sigma_star zd zs

Given a set of parameters regarding an extragalactic microlensing scheme, this
program computes the probability of magnification in a given range.
    
positional arguments:
  mu_t                  Value of the tangential macro-magnification.
  mu_r                  Value of the radial macro-magnification.
  sigma_star            Surface mass density of microlenses [Msun/pc2].
  zd                    Redshift at the lens plane (cluster).
  zs                    Redshift at the source plane.

optional arguments:
  -h, --help            show this help message and exit
  --mu1 mu1             Minimum magnification to display the pdf.
  --mu2 mu2             Maximum magnification to display the pdf.
  --dir [DIR]           Directory where the results will be stored.
  --plot plot           If "True", plot and save the pdf.
  --save save           If "True", save the pdf in a file.
  --extension extension
                        If save, extension in which the data is saved (txt,
                        fits, h5).
    
Contact: palencia@ifca.unican.es / jpalenciasainz@gmail.com
```
Usage example:
```
$ python M_SMiLe.py -600 2 5 1 1.7 --dir /foo/bar/test/ --save False --mu2 1000
```
* As a python class.

Any python program can import the class microlenses from `M_SMiLe.py`.

Once we have imported the class we can create and instance of an object and call its methods to save the data in different files, generate plots, or directly get **numpy** arrays with the value of the magnification probability.

```python
# Import the microlenses class from M_SMiLe.py
from M_SMiLe import microlenses

# Create an object of the class microlenses with the desired inputs
microlens = microlenses(mu_t=200, mu_r=4, sigma_star=12.4, zs=1.3, zd=0.7, mu1=1e-3, mu2=1e5)

# Get magnification probability per logaritmic bin
pdf, log_mu = microlens.get_pdf()

# Save data in a file (h5, txt, fits). Can choose another path.
microlens.save_data(extension='fits')

# Save a plot.
microlens.plot(save_pic=True)
```
# Output examples
[Neg_parity_High_sigma.pdf](https://github.com/ChemaPalencia/M_SMiLe/files/12039962/Neg_parity_High_sigma.pdf)

[Neg_parity_Low_sigma.pdf](https://github.com/ChemaPalencia/M_SMiLe/files/12039963/Neg_parity_Low_sigma.pdf)

[Pos_parity_High_sigma.pdf](https://github.com/ChemaPalencia/M_SMiLe/files/12039964/Pos_parity_High_sigma.pdf)

[Pos_parity_Low_sigma.pdf](https://github.com/ChemaPalencia/M_SMiLe/files/12039965/Pos_parity_Low_sigma.pdf)

# License

This project is licensed under the **MIT License**. Feel free to use and modify the code according to the terms specified in the license.

# Contact

If you have any questions or inquiries regarding this code or its usage, please contact palencia@ifca.unican.es or jpalenciasainz@gmail.com

We hope this code proves to be useful in your research and exploration of magnification probability of high redshift stars by galaxy clusters. Happy computing!

