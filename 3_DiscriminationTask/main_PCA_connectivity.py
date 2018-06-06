
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


#### Supplementary code for the paper: 
#### "Linking connectivity, dynamics and computations in recurrent neural networks"
#### F. Mastrogiuseppe and S. Ostojic (2018)

#### CODE 3: sample rank-one network for the Go-Nogo discrimination task (related to Fig. 3 B-C-E-F)
#### The code is composed of three main scripts, to be run in the order:
#### main_simulate.py  -  generates a network and the activity in response to Go and Nogo stimulus
#### main_plot.py  -  plots single neuron and population activity
#### main_PCA_connectivity.py  -  computes the first PC axis for Go and Nogo trials, and correlates
####                              them with the average pair connectivity

#### Note that the Data/ folder is empty to begin; the code main_simulate.py needs to be run  
#### at least once


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import matplotlib.pyplot as plt
import numpy as np

import fct_simulations as sim
import fct_facilities as fac


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Load data

doCompute = 1

path_data = 'Data/'

structure = fac.Retrieve('structure.p', path_data)
activation = fac.Retrieve('activation.p', path_data)
connectivity = fac.Retrieve('average_connectivity.p', path_data)

N = activation.shape[2]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Compute PC axis

# We compute the PC axis separately from the Go and the Nogo trials

XA = activation[0,:,:]
XB = activation[1,:,:]

# Z-score

YA = XA - np.outer (np.ones(XA.shape[0]), np.mean(XA, 0))   # Remove the mean
#YA = YA / np.outer (np.ones(YA.shape[0]), np.std(YA, 0))   # Divide by std

YB = XB - np.outer (np.ones(XB.shape[0]), np.mean(XB, 0)) 
#YB = YB / np.outer (np.ones(YB.shape[0]), np.std(YB, 0)) 

# Compute PC

CA = np.dot(YA.T, YA) / (YA.shape[0] - 1)   # Compute covariance matrix
CB = np.dot(YB.T, YB) / (YB.shape[0] - 1) 

eigA = np.linalg.eig(CA)   # Diagonalize it
VA = eigA[1].real 
lambdasA = eigA[0].real

eigB = np.linalg.eig(CB)
VB = eigB[1].real 
lambdasB = eigB[0].real 

# Order the PC

lambdasA = lambdasA [np.flipud(np.argsort(lambdasA))]
VA = VA[:,np.flipud(np.argsort(lambdasA))]

lambdasB = lambdasB [np.flipud(np.argsort(lambdasB))]
VB = VB[:,np.flipud(np.argsort(lambdasB))]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Compare with average connectivity

pcaA_correlation = np.outer(VA[:,0], VA[:,0]) # Consider the first PC axis
pcaB_correlation = np.outer(VB[:,0], VB[:,0])

Nsample = 20000 # Subsample pairs of neurons for computing the correlation coefficient
idx = np.random.randint(0, N**2, Nsample)

Nsample_plot = 1000 # Subsample pairs of neurons for plotting
idx_plot = np.random.randint(0, N**2, Nsample_plot)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Plot

fac.SetPlotParams()
dashes = [3,3]

color_Go = '#1C63A9'
color_Nogo = '#009999'


# Correlation between connectivity and PCA derived from Go trials

fg = plt.figure()
ax0 = plt.axes(frameon=True)

fit = np.polyfit( pcaA_correlation.flatten()[idx] , connectivity.flatten()[idx], 1)
x_range = np.linspace(-1,1,100)
corr_coeff = np.corrcoef(pcaA_correlation.flatten()[idx] , connectivity.flatten()[idx])

plt.plot(pcaA_correlation.flatten()[idx_plot], connectivity.flatten()[idx_plot], 'o', color = color_Go, alpha = 0.5)

line, = plt.plot(x_range, fit[1] + x_range*fit[0], '--', color='#CC0000', alpha = 1, label=r'$\rho='+str(round(corr_coeff[0,1],2))+'$' )
line.set_dashes(dashes)

plt.xlim(-0.003, 0.003)
plt.ylim(-0.02, 0.02)

plt.xticks([-0.003, 0, 0.003])

plt.xlabel(r'PC 1 correlation')
plt.ylabel(r'Pair connectivity')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)
plt.legend(loc=2)

plt.savefig('connectivity_A.pdf')
plt.show()


# Correlation between connectivity and PCA derived from Nogo trials

fg = plt.figure()
ax0 = plt.axes(frameon=True)

fit = np.polyfit( pcaB_correlation.flatten()[idx] , connectivity.flatten()[idx], 1)
x_range = np.linspace(-1,1,100)
corr_coeff = np.corrcoef(pcaB_correlation.flatten()[idx] , connectivity.flatten()[idx])

plt.plot(pcaB_correlation.flatten()[idx_plot], connectivity.flatten()[idx_plot], 'o', color = color_Nogo, alpha = 0.5)

line, = plt.plot(x_range, fit[1] + x_range*fit[0], '--', color='#CC0000', alpha = 1, label=r'$\rho='+str(round(corr_coeff[0,1],2))+'$' )
line.set_dashes(dashes)

plt.xlim(-0.003, 0.003)
plt.ylim(-0.02, 0.02)

plt.xticks([-0.003, 0, 0.003])

plt.xlabel(r'PC 1 correlation')
plt.ylabel(r'Pair connectivity')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)
plt.legend(loc=2)

plt.savefig('connectivity_B.pdf')
plt.show()

#
