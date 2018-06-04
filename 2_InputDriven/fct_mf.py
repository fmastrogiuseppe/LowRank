import numpy as np
import scipy

from fct_integrals import *

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Solve mean-field equations

### Non-trivial solutions, solved through iteration

def SolveStatic (y0, g, VecPar, tolerance = 1e-10, backwards = 1):  # y[0]=mu, y[1]=Delta0

    # The variable y contains the mean-field variables mu, delta0 and K
    # Note that, for simplicity, only delta0 and one first-order statistics (kappa) get iterated
    # The variable backwards can be set to (-1) to invert the flow of iteration and reach unstable solutions

    again = 1
    y = np.array(y0)
    y_new = np.ones(3)
    eps = 0.2

    Mm, Mn, Mi, Sim, Sin, Sini, Sip = VecPar
    Sii = np.sqrt( (Sini/Sin)**2 + Sip**2 )

    while (again==1):

        # Take a step

        mu = Mm * y[2] + Mi
        new1 = g*g * PhiSq(mu, y[1]) + Sim**2 * y[2]**2 + Sii**2
        new2 =  Mn * Phi(mu, y[1]) + Sini * Prime(mu, y[1])

        y_new[0] = Mm * new2 + Mi
        y_new[1] = (1-eps)*y[1] + eps*new1
        y_new[2] = (1-backwards*eps)*y[2] + backwards*eps*new2

        # Stop if the variables converge to a number, or zero
        # If it becomes nan, or explodes
        
        if( np.fabs(y[1]-y_new[1]) < tolerance*np.fabs(y[1]) and np.fabs(y[2]-y_new[2]) < tolerance*np.fabs(y[2]) ):
            again = 0

        if( np.fabs(y[1]-y_new[1]) < tolerance and np.fabs(y[2]-y_new[2]) < tolerance ):
            again = 0

        if np.isnan(y_new[0]) == True:
            again = 0
            y_new = [0,0,0]

        if( np.fabs(y[2])> 1/tolerance ):
            again = 0
            y_new = [0,0,0]
    
        y[0] = y_new[0]
        y[1] = y_new[1]
        y[2] = y_new[2]

    return y_new
