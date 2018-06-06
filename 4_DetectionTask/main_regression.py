
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


#### Supplementary code for the paper: 
#### "Linking connectivity, dynamics and computations in recurrent neural networks"
#### F. Mastrogiuseppe and S. Ostojic (2018)

#### CODE 4: sample rank-one network for the Go-Nogo detection (related to Fig. 4 B-C-D-E)
#### The code is composed of three main scripts, to be run in the order:
#### main_simulate.py  -  generates a network and the activity in response to four stimulus intensities,
####                      for Nic different noise realizations
#### main_plot.py  -  plots single neuron and population activity
#### main_regression.py  -  computes the regression axis for input and choice

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
readout = fac.Retrieve('readout.p', path_data)

ParVec = fac.Retrieve('ParVec.p', path_data)
c_values = fac.Retrieve('c_values.p', path_data)
sigma = fac.Retrieve('sigma.p', path_data)

t = fac.Retrieve('t.p', path_data)
t1 = fac.Retrieve('t1.p', path_data)
t2 = fac.Retrieve('t2.p', path_data)

Sii, Siw, rhom, rhon = fac.Retrieve('ParVec.p', path_data)

N = activation.shape[3]
I = structure[1,:]
m = structure[0,:] + ParVec[2]*structure[2,:]
Nic = activation.shape[1]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Linear regression

# Regresses the network activation onto the 'input' and 'choice' variables
# The input strength is taken equal to the average value c
# The choice variable is taken equal to the readout z at the end of stimulus presentation

beta = np.zeros (( 2, len(t), N ))     # Regression coefficients for choice & input
beta_time_ind = np.zeros (( 2, N ))    # Time-independent regression coefficients

# Build matrix F

F = np.zeros (( 2, len(c_values) * Nic))

for i, c in enumerate(c_values):

	F[0, i * Nic : (i+1) * Nic] = readout[i, :, -1]
	F[1, i * Nic : (i+1) * Nic] = c

# Re-arrange the activation matrix	

X = np.zeros (( len(c_values) * Nic, len(t), N ))

for i, c in enumerate(c_values):

	X[i * Nic : (i+1) * Nic, :, :] = activation[i, :, :, :]

### Compute regression coefficients

for j in range(len(t)):
	for i in range(N):

		ls_matrix = np.dot( np.linalg.inv ( np.dot(F, F.T) ), F )

		beta[:, j, i] = np.dot( ls_matrix, X[:, j, i] ) 

# Compute time-independent regression coefficients by selecting the time point where the norm
# of beta is maximized

norm = np.zeros(( len(t) ))

for j in range(len(t)):

	norm[j] = np.linalg.norm(beta[0, j, :])**2 + np.linalg.norm(beta[1, j, :])**2

beta_time_ind [0,:] = beta[0, np.argmax(norm), :]
beta_time_ind [1,:] = beta[1, np.argmax(norm), :]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Plot

fac.SetPlotParams()
dashes = [3,3]

fac.SetPlotDim(1.45, 1.35)


# Scatter plot of the correlation for input (x axis) and choice (y axis)

fg=plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot([-10, 10], [0,0], color = 'k', linewidth = 0.7)
plt.plot([0,0], [-40, 40], color = 'k', linewidth = 0.7)
plt.plot(beta_time_ind[1,:], beta_time_ind[0,:], 'o', color = '0.7', alpha = 0.5)

plt.axis('off')

plt.savefig('regression.pdf')
plt.show()


# Correlation matrix: input and choice vs m and I

fac.SetPlotDim(0.78, 0.72)

corr_matrix = np.zeros (( 2, 2 ))

corr_matrix[0,0] = np.corrcoef( m, beta_time_ind[0,:] )[0,1]
corr_matrix[0,1] = np.corrcoef( m, beta_time_ind[1,:] )[0,1]
corr_matrix[1,0] = np.corrcoef( I, beta_time_ind[0,:] )[0,1]
corr_matrix[1,1] = np.corrcoef( I, beta_time_ind[1,:] )[0,1]

fg=plt.figure()
ax0 = plt.axes(frameon=True)

plt.imshow(corr_matrix, vmin = 0, vmax = 1, extent=(0, 10, 0, 1), cmap='Greys', origin='lower', aspect='auto', interpolation = 'nearest')

ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')

plt.xticks([])
plt.yticks([])

plt.grid('off')
plt.colorbar(ticks = [0, 1])

plt.savefig('corr_matrix.pdf')
plt.show()

#
