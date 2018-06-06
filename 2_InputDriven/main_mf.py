#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


#### Supplementary code for the paper: 
#### "Linking connectivity, dynamics and computations in recurrent neural networks"
#### F. Mastrogiuseppe and S. Ostojic (2018)

#### CODE 2: input-driven activity in rank-one networks: DMF theory (related to Fig. 2D right)
#### This code computes the DMF solutions for increasing values of the input along n
#### The overlap direction is defined along the unitary direction (rho = 0, see Methods)
#### Within the DMF theory, activity is then described in terms of mean (mu), overlap (kappa) and variance (delta) of x
#### The input contains no component along m (Simi = 0, see Methods)

#### Note that the Data/ folder is empty to begin; this code needs to be run with the flag doCompute = 1
#### at least once


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import matplotlib.pyplot as plt
import numpy as np

import fct_facilities as fac
import fct_mf as mf
import fct_integrals as integ


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Parameters

### Set parameters

Mm = 3.5      # Mean of m
Mn = 1.       # Mean of n
Mi = 0.       # Mean of I

Sim = 1.      # Std of m
Sin = 1.      # Std of n
Sip = 1.      # Std of input orthogonal to m and n, along h (see Methods)

g = 0.8

Sini_min = 1e-4
Sini_max = 2.
Sini_values = np.linspace(Sini_min, Sini_max, 400)   # Input correlating with n along x2 (see Methods)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Compute

doCompute = 1

path_here = 'Data/'

if doCompute == True:

    # For this choice of parameters no chaotic solutions are admitted, we compute only stationary ones
    # The DMF equations admit at maximum three different solutions, which can be reached starting from different i.c.

    delta0_s = np.zeros(( 3, len(Sini_values) ))
    mu_s = np.zeros(( 3, len(Sini_values) ))
    K_s = np.zeros(( 3, len(Sini_values) ))

    # Initial conditions for iteration

    ics_0 = [5., 5., 5.]
    ics_1 = [-5., 5., -5,]
    ics_2 = [0., 1e-2, 0.]

    for i, Sini in enumerate(Sini_values):

        ParVec = [Mm, Mn, Mi, Sim, Sin, Sini, Sip]

        print Sini

        ### Compute POSITIVE solutions
        
        mu_s[0,i], delta0_s[0,i], K_s[0,i] = mf.SolveStatic ( ics_0, g, ParVec)
        ics_0 = [ mu_s[0,i], delta0_s[0,i], K_s[0,i] ]
        
        ### Compute NEGATIVE solutions

        mu_s[1,i], delta0_s[1,i], K_s[1,i] = mf.SolveStatic ( ics_1, g, ParVec)
        ics_1 = [ mu_s[1,i], delta0_s[1,i], K_s[1,i] ]

        ics_2 = [Mm*K_s[1,i]+0.1 + Mi, 0.1*delta0_s[1,i], K_s[1,i]+0.1]

        ### Compute CENTRAL solution

        mu_s[2,i], delta0_s[2,i], K_s[2,i] = mf.SolveStatic ( ics_2, g, ParVec, backwards = -1. )

    # Store

    fac.Store( K_s, 'K_s.p', path_here)
    fac.Store( mu_s, 'mu_s.p', path_here)
    fac.Store( delta0_s, 'delta0_s.p', path_here)

else:

    # Retrieve

    K_s = fac.Retrieve('K_s.p', path_here)
    mu_s = fac.Retrieve('mu_s.p', path_here)
    delta0_s = fac.Retrieve('delta0_s.p', path_here)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Plot

fac.SetPlotParams()
dashes = [3, 3] 

color_s = '#4872A1'


# Kappa
# Plot the negative and central branch of the solution only when they are not equal to the positive one

fg=plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot((Sini_values/Sin), K_s[0,:], color=color_s)

plt.plot((Sini_values/Sin)[mu_s[1,:]<0], K_s[1,:][mu_s[1,:]<0], color=color_s)

line, = plt.plot((Sini_values/Sin)[mu_s[1,:]<0], K_s[2,:][mu_s[1,:]<0], color=color_s, ls='--', alpha=0.5)
line.set_dashes(dashes)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.xlim(0, max(Sini_values))
plt.xticks([0, 1., 2.])

plt.ylim(-1.5, 1.5)
plt.yticks([-1.5, 0, 1.5])

#plt.legend(loc=1)
plt.xlabel('Input strength on $n_{\perp}$')
plt.ylabel('Activity along $m$ ($\kappa$)')
plt.savefig('k.pdf')

plt.show()

#
