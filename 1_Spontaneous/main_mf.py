
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


#### Supplementary code for the paper: 
#### "Linking connectivity, dynamics and computations in recurrent neural networks"
#### F. Mastrogiuseppe and S. Ostojic (2018)

#### CODE 1a: spontaneous activity in rank-one networks: DMF theory (related to Fig. 1C)
#### This code computes the DMF solutions (and their stability) for increasing values of the random strength g
#### The overlap direction is defined along the unitary direction (rho = 0, see Methods)
#### Within the DMF theory, activity is then described in terms of mean (mu) and variance (delta) of x


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import matplotlib.pyplot as plt
import numpy as np

import fct_integrals as integ
import fct_facilities as fac
import fct_mf as mf


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Parameters

Mm = 1.1        # Mean of m
Mn = 2.         # Mean of n
Sim = 1.        # Std of m
Sin = 1.        # Std of n

g_min = 1e-4
g_max = 4.
g_values = np.linspace(g_min, g_max, 200)        # Random strength


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Compute

# We solve separately the DMF equations corresponding to stationary and chaotic states
# The DMF equations admit at maximum three different solutions, which can be reached starting from different i.c.
# We compute two variance values: the population variance delta0 and the individual one delta0I (see Methods)
# Within chaotic phases, the stationary fraction of the variance deltainf is computed as well (see Methods)

doCompute = 0

path_here = 'Data/'

ParVec = [Mm, Mn, Sim, Sin]

if doCompute == True:

    # Stationary solutions
    delta0_s = np.zeros(( 3, len(g_values) ))
    delta0I_s = np.zeros(( 3, len(g_values) ))
    mu_s = np.zeros(( 3, len(g_values) ))

    # Chaotic solutions
    delta0_c = np.zeros(( 3, len(g_values) ))
    delta0I_c = np.zeros(( 3, len(g_values) ))
    deltainf_c = np.zeros(( 3, len(g_values) ))
    deltainfI_c = np.zeros(( 3, len(g_values) ))
    mu_c = np.zeros(( 3, len(g_values) ))

    # Eigenspectra of stationary solutions
    radius = np.zeros(( 3, len(g_values) ))
    outlier = np.zeros(( 3, len(g_values) ))

    # Initial conditions for iteration
    ics_0 = [50., 50.]
    ics_1 = [-50., 50.]
    flag_0 = 0
    flag_1 = 0

    for i, g in enumerate(g_values):

        print g

        ### Compute POSITIVE solutions
        
        mu_s[0,i], delta0_s[0,i] = mf.SolveStatic ( ics_0, g, ParVec)
        delta0I_s[0,i] = delta0_s[0,i] - ( Sim*Mn*mf.Phi(mu_s[0,i], delta0_s[0,i]) )**2
        radius[0,i] = g*np.sqrt(integ.PrimeSq(mu_s[0,i], delta0_s[0,i]))

        ics_0 = [ mu_s[0,i], delta0_s[0,i]]

        if flag_0 == 0: icc_0 = [ mu_s[0,i], 1.1*delta0_s[0,i], 0.9*delta0_s[0,i]] 

        if (radius[0,i]>1):

            flag_0 = 1

            mu_c[0,i], delta0_c[0,i], deltainf_c[0,i] = mf.SolveChaotic ( icc_0, g, ParVec)
            delta0I_c[0,i] = delta0_c[0,i] - ( Sim*Mn*mf.Phi(mu_c[0,i], delta0_c[0,i]) )**2
            deltainfI_c[0,i] = deltainf_c[0,i] - ( Sim*Mn*mf.Phi(mu_c[0,i], delta0_c[0,i]) )**2
            icc_0 = [ mu_c[0,i], 1.3*delta0_c[0,i], 0.9*delta0_c[0,i]]
        
        ### Compute NEGATIVE solutions

        mu_s[1,i], delta0_s[1,i] = mf.SolveStatic ( ics_1, g, ParVec)
        delta0I_s[1,i] = delta0_s[1,i] - ( Sim*Mn*mf.Phi(mu_s[1,i], delta0_s[1,i]) )**2
        radius[1,i] = g*np.sqrt(integ.PrimeSq(mu_s[1,i], delta0_s[1,i]))

        ics_1 = [ mu_s[1,i], delta0_s[1,i]]

        if flag_1 == 0: icc_1 = [ mu_s[1,i], delta0_s[1,i], 0.9*delta0_s[1,i]] 

        if (radius[1,i]>1):

            flag_1 = 1

            mu_c[1,i], delta0_c[1,i], deltainf_c[1,i] = mf.SolveChaotic ( icc_1, g, ParVec)
            delta0I_c[1,i] = delta0_c[1,i] - ( Sim*Mn*mf.Phi(mu_c[1,i], delta0_c[1,i]) )**2
            deltainfI_c[1,i] = deltainf_c[1,i] - ( Sim*Mn*mf.Phi(mu_c[1,i], delta0_c[1,i]) )**2
            icc_1 = [ mu_c[1,i], 1.3*delta0_c[1,i], 0.9*delta0_c[1,i]]

        ### Compute CENTRAL solutions (characterized by my = deltainf = 0)

        delta0_s[2,i] = mf.SolveStatic_0(g)
        delta0_c[2,i] = mf.SolveChaotic_0(g)
        delta0I_s[2,i] = delta0_s[2,i] - ( Sim*Mn*mf.Phi(mu_s[2,i], delta0_s[2,i]) )**2
        delta0I_c[2,i] = delta0_c[2,i] - ( Sim*Mn*mf.Phi(mu_c[2,i], delta0_c[2,i]) )**2
        radius[2,i] = g*np.sqrt(integ.PrimeSq(mu_s[2,i], delta0_s[2,i]))

        ### Compute the stability outliers
        
        outlier[0,i] = mf.EigStationary ( g, ParVec, [mu_s[0,i], delta0_s[0,i]] )
        outlier[1,i] = mf.EigStationary ( g, ParVec, [mu_s[1,i], delta0_s[1,i]] )
        outlier[2,i] = mf.EigStationary ( g, ParVec, [mu_s[2,i], delta0_s[2,i]] )

    # Store

    fac.Store( mu_s, 'mu_s.p', path_here)
    fac.Store( delta0_s, 'delta0_s.p', path_here)
    fac.Store( delta0I_s, 'delta0I_s.p', path_here)
    fac.Store( mu_c, 'mu_c.p', path_here)
    fac.Store( delta0_c, 'delta0_c.p', path_here)
    fac.Store( deltainf_c, 'deltainf_c.p', path_here)
    fac.Store( delta0I_c, 'delta0I_c.p', path_here)
    fac.Store( deltainfI_c, 'deltainfI_c.p', path_here)
    fac.Store( radius, 'radius.p', path_here)
    fac.Store( outlier, 'outlier.p', path_here)

else:

    # Retrieve

    mu_s = fac.Retrieve('mu_s.p', path_here)
    delta0_s = fac.Retrieve('delta0_s.p', path_here)
    delta0I_s = fac.Retrieve('delta0I_s.p', path_here)
    mu_c = fac.Retrieve('mu_c.p', path_here)
    delta0_c = fac.Retrieve('delta0_c.p', path_here)
    deltainf_c = fac.Retrieve( 'deltainf_c.p', path_here)
    delta0I_c = fac.Retrieve('delta0I_c.p', path_here)
    deltainfI_c = fac.Retrieve( 'deltainfI_c.p', path_here)
    radius = fac.Retrieve('radius.p', path_here)
    outlier = fac.Retrieve('outlier.p', path_here)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Plot
# We plot K = Mn*phi as first-order statistics and individual variances as second-order statistics

K_s = mu_s / Mm
K_c = mu_c / Mm

gC = 1.796 # Value of the instability to chaotic activity
gBB = 2.136 # Value where the two bistable chaotic solutions collapse to zero

#

fac.SetPlotParams()
dashes = [3, 3]

color_scs = '0.6'
color_s = '#4872A1'
color_c1 = '#C44343'
color_c2 = '#FF9999'
color_sim0 = '0'
color_sim1 = '0.5'

thr = 1e-10 # Plot chaotic solutions only when they are non-zero


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Plot K

fg = plt.figure()
ax0 = plt.axes(frameon = True)

# Stationary

line, = plt.plot(g_values, K_s[2,:], color = color_s, ls='--', alpha = 0.5)
line.set_dashes(dashes)

plt.plot(g_values[g_values<gC], K_s[0,:][g_values<gC], color = color_s, label = 'Static')
plt.plot(g_values[g_values<gC], K_s[1,:][g_values<gC], color = color_s)

plt.plot(g_values[g_values>gC], K_s[0,:][g_values>gC], color = color_s, alpha = 0.4)
plt.plot(g_values[g_values>gC], K_s[1,:][g_values>gC], color = color_s, alpha = 0.4)

# Chaotic

line, = plt.plot(g_values[delta0_c[2,:]>thr], K_c[2,:][delta0_c[2,:]>thr], color=color_c1, ls='--', alpha = 0.5)
line.set_dashes(dashes)

plt.plot(g_values[delta0_c[0,:]>thr], K_c[0,:][delta0_c[0,:]>thr], color = color_c1, label = 'Chaotic')
plt.plot(g_values[delta0_c[1,:]>thr], K_c[1,:][delta0_c[1,:]>thr], color = color_c1)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.xlim(0, max(g_values))

plt.legend(loc=1)

plt.xlabel('Random strength $g$')
plt.ylabel('Activity along $m$ ($\kappa$)')
plt.savefig('k.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Plot delta0I

fg = plt.figure()
ax0 = plt.axes(frameon=True)

# Stationary

line, = plt.plot(g_values, delta0I_s[2,:], color = color_s, ls='--', alpha = 0.5)
line.set_dashes(dashes)

plt.plot(g_values[g_values<gC], delta0I_s[0,:][g_values<gC], color = color_s)

plt.plot(g_values[g_values>gC], delta0I_s[0,:][g_values>gC], color = color_s, alpha = 0.4)

# Chaotic

line, = plt.plot(g_values[delta0_c[2,:]>thr], delta0I_c[2,:][delta0_c[2,:]>thr] - deltainfI_c[2,:][delta0_c[2,:]>thr], color = color_c1, ls = '--', alpha = 0.5)
line.set_dashes(dashes)

plt.plot(g_values[delta0_c[0,:]>thr], deltainfI_c[0,:][delta0_c[0,:]>thr], color = color_c2, label='$\Delta_0^I-\Delta_T^I$')

plt.plot(g_values[delta0_c[0,:]>thr], delta0I_c[0,:][delta0_c[0,:]>thr] - deltainfI_c[0,:][delta0_c[0,:]>thr], color = color_c1, label='$\Delta_T^I$')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.xlim(0, max(g_values))
plt.ylim(-0.05,8)
plt.yticks([0, 2, 4, 6, 8])

plt.legend(loc=2)

plt.xlabel('Random strength $g$')
plt.ylabel('Individual variance')
plt.savefig('delta.pdf')
plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Stability eigenspectra

fg = plt.figure()
ax0 = plt.axes(frameon=True)

# Radius of non-trivial stationary solutions
plt.plot(g_values[g_values<gC], radius[0,:][g_values<gC], color='0.5', label = 'Radius')
line, = plt.plot(g_values[g_values>gC], radius[0,:][g_values>gC], color='0.5')
line.set_dashes(dashes)

# Outlier of non-trivial stationary solutions
plt.plot(g_values[outlier[0,:]>radius[0,:]], outlier[0,:][outlier[0,:]>radius[0,:]], color=color_s, label = 'Outlier')

plt.legend(loc=2)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.xlim(0, max(g_values))
plt.ylim(-0.1,2)

plt.xlabel('Random strength $g$')
plt.ylabel('Stability eigenvalue')
plt.savefig('spectrum.pdf')
plt.show()

#
