import matplotlib.pyplot as plt
import numpy as np

from fct_integrals import *

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Solve mean-field equations

### Solve the DMF equation through iteration

def SolveStatic (y0, g, VecPar, backwards = 1, tolerance = 1e-14):

    # The variable y contains the mean-field variables K1, K2 and delta0
    # Note that mu = 0 (see Methods)
    # The variable backwards can be set to (-1) to invert the flow of iteration and reach unstable solutions

    again = 1
    y = np.array(y0)
    y_new = np.ones(3)
    eps = .1

    rho, Si = VecPar

    while (again==1):

        # Take a step

        new0 =  y[0] * rho**2 * Prime(0, y[2])
        new1 =  y[1] * rho**2 * Prime(0, y[2])
        new2 = g*g * PhiSq(0, y[2]) + Si**2 * (y[0]**2+y[1]**2)

        y_new[0] = (1-eps*backwards)*y[0] + eps*backwards*new0
        y_new[1] = (1-eps*backwards)*y[1] + eps*backwards*new1
        y_new[2] = (1-eps)*y[2] + eps*new2

        # Stop if the variables converge to a number, or zero
        # If it becomes nan, or explodes

        if( np.fabs(y[2]-y_new[2]) < np.fabs(y[2])*tolerance ):
            again = 0

        if( np.fabs(y[2]) < tolerance ):
            again = 0

        if( np.fabs(y[0])> 1/tolerance ):
            again = 0
            y_new = [0, 0, 0]

        y[0] = y_new[0]
        y[1] = y_new[1]
        y[2] = y_new[2]

    return y_new


def SolveChaotic (y0, g, VecPar, backwards = 1, tolerance=1e-8):

    # The variable y contains the mean-field variables K1, K2, delta0 and deltainf
    # Note that mu = 0 (see Methods)
    # The variable backwards can be set to (-1) to invert the flow of iteration and reach unstable solutions

    again = 1
    y = np.array(y0)
    y_new = np.ones(4)
    eps = .1

    rho, Si = VecPar

    while (again==1):

        # Take a step

        new0 =  y[0] * rho**2 * Prime(0, y[2])
        new1 =  y[1] * rho**2 * Prime(0, y[2])
        new2 = np.sqrt( np.max ( [0, y[3]**2 +2* (g*g* ( PrimSq(0, y[2]) - IntPrimPrim(0, y[2], y[3]))  + Si**2 * (y[0]**2+y[1]**2) * (y[2] - y[3]) )  ] ) )
        new3 = g*g*IntPhiPhi(0, y[2], y[3]) + Si**2 * (y[0]**2+y[1]**2)

        y_new[0] = (1-eps*backwards)*y[0] + eps*backwards*new0
        y_new[1] = (1-eps*backwards)*y[1] + eps*backwards*new1
        y_new[2] = (1-eps)*y[2] + eps*new2
        y_new[3] = (1-eps)*y[3] + eps*new3

        # Stop if the variables converge to a number, or zero
        # If it becomes nan, or explodes

        if( ( np.fabs(y[2]-y_new[2]) < np.fabs(y[2])*tolerance ) and  ( np.fabs(y[0]-y_new[0]) < np.fabs(y[0])*tolerance ) ):
            again = 0 

        if( ( np.fabs(y[2]-y_new[2]) < np.fabs(y[2])*tolerance ) and  ( np.fabs(y[0]-y_new[0]) < tolerance ) ):
            again = 0

        if( np.fabs(y[2]) < 1e-16  ):
            again = 0

        if( np.fabs(y[0]) > 1/tolerance ):
            again = 0
            y_new = [0, 0, 0, 0]

        if (y_new[3] > y_new[2]):
            again = 0

        y[0] = y_new[0]
        y[1] = y_new[1]
        y[2] = y_new[2]
        y[3] = y_new[3]

    return y_new
