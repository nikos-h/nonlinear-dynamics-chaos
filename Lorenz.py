############################################################
# This file contains related functions for integrating and reducing
# Lorenz system.
# 
# please fill out C2, velocity(), stabilityMatrix(),
# integrator_with_jacob(), reduceSymmetry(), case3 and case4.
############################################################

import numpy as np
import matplotlib.pyplot as plt
import Rossler
from mpl_toolkits.mplot3d import Axes3D

from numpy.random import rand
from scipy.integrate import odeint

# G_ means global
G_sigma = 10.0
G_rho = 28.0
G_b = 8.0/3.0

# complete the definition of C^{1/2} operation matrix for Lorenz
# system. 
C2 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [ 0, 0, 1]
        ])


def velocity(stateVec, t):
    """
    return the velocity field of Lorentz system.
    stateVec : the state vector in the full space. [x, y, z]
    t : time is used since odeint() requires it. 
    """
    
    x = stateVec[0]
    y = stateVec[1]
    z = stateVec[2]
    
    # complete the flowing 3 lines.
    vx =  G_sigma*(y - x)
    vy = G_rho*x - y - x*z
    vz = x*y - G_b*z

    return np.array([vx, vy, vz])

def stabilityMatrix(stateVec):
    """
    return the stability matrix at a state point.
    stateVec: the state vector in the full space. [x, y, z]
    """
    
    x = stateVec[0]; y = stateVec[1]; z = stateVec[2];
    # fill out the following matrix.
    stab = np.array([
            [-G_sigma, G_sigma, 0],
            [G_rho, -1, -x],
            [y, x, -G_b]
            ])
    
    return stab

def integrator(init_x, dt, nstp):
    """
    The integator of the Lorentz system.
    init_x: the intial condition
    dt : time step
    nstp: number of integration steps.
    
    return : a [ nstp x 3 ] vector 
    """

    state = odeint(velocity, init_x, np.arange(0, dt*nstp, dt))
    return state

def integrator_with_jacob(init_x, dt, nstp):
    """
    integrate the orbit and the Jacobian as well. The meaning 
    of input parameters are the same as 'integrator()'.
    
    return : 
            state: a [ nstp x 3 ] state vector 
            Jacob: [ 3 x 3 ] Jacobian matrix
    """

    # Please fill out the implementation of this function.
    # You can go back to the previous homework to see how to
    # integrate state and Jacobian at the same time.
    
    
    state = integrator(init_x, dt, nstp)
    Jacob = Jacobian(init_x, dt, nstp)
    
    return state, Jacob

def reduceSymmetry(states):
    """
    reduce C^{1/2} symmetry of Lorenz system by invariant polynomials.
    (x, y, z) -> (u, v, z) = (x^2 - y^2, 2xy, z)
    
    states: trajectory in the full state space. dimension [nstp x 3]
    return: states in the invariant polynomial basis. dimension [nstp x 3]
    """
    
    m, n = states.shape
    reducedStates = np.zeros([m, n])

    u = lambda x, y: x*x - y*y
    v = lambda x, y: 2.0*x*y

    for i in range(m):
        reducedStates[i] = [u(states[i][0], states[i][1]),
                                v(states[i][0], states[i][1]),
                                      states[i][2]]
    
    
    return reducedStates

def JacobianVelocity(sspJacobian, dt):
    """
    Velocity function for the Jacobian integration

    Inputs:
    sspJacobian: (d+d^2)x1 dimensional state space vector including both the
                 state space itself and the tangent space
    t: Time. Has no effect on the function, we have it as an input so that our
       ODE would be compatible for use with generic integrators from
       scipy.integrate

    Outputs:
    velJ = (d+d^2)x1 dimensional velocity vector
    """

    ssp = sspJacobian[0:3]  # First three elements form the original state
                            # space vector
    J = sspJacobian[3:].reshape((3, 3))  # Last nine elements corresponds to
                                         # the elements of Jacobian.
    
    velJ = np.zeros(np.size(sspJacobian))  # Initiate the velocity vector as a
                                           # vector of same size with
                                           # sspJacobian
    velJ[0:3] = velocity(ssp, dt)
    #Last dxd elements of the velJ are determined by the action of
    #stability matrix on the current value of the Jacobian:
    velTangent = np.dot(stabilityMatrix(ssp), J)  # Velocity matrix for
                                                  #  the tangent space
    velJ[3:] = np.reshape(velTangent, 9)  # Another use of numpy.reshape, here
                                          # to convert from dxd to d^2
    return velJ

def Jacobian(ssp, dt, nstp):
    """
    Jacobian function for the trajectory started on ssp, evolved for time t

    Inputs:
    ssp: Initial state space point. dx1 NumPy array: ssp = [x, y, z]
    t: Integration time
    Outputs:
    J: Jacobian of trajectory f^t(ssp). dxd NumPy array
    """
    #CONSTRUCT THIS FUNCTION
    #Hint: See the Jacobian calculation in CycleStability.py
    #J = None
    Jacobian0 = np.identity(3)  # COMPLETE THIS LINE. HINT: Use np.identity(DIMENSION)
    #Initial condition for Jacobian integral is a d+d^2 dimensional matrix
    #formed by concatenation of initial condition for state space and the
    #Jacobian:
    sspJacobian0 = np.zeros(3 + 3 ** 2)  # Initiate
    sspJacobian0[0:3] = ssp  # First 3 elemenets
    sspJacobian0[3:] = np.reshape(Jacobian0, 9)  # Remaining 9 elements
    tInitial = 0  # Initial time
    tFinal = dt*nstp  # Final time
    Nt = nstp  # Number of time points to be used in the integration

    tArray = np.linspace(tInitial, tFinal, Nt)  # Time array for solution

    sspJacobianSolution = odeint(JacobianVelocity, sspJacobian0, tArray)

    xt = sspJacobianSolution[:, 0]  # Read x(t)
    yt = sspJacobianSolution[:, 1]  # Read y(t)
    zt = sspJacobianSolution[:, 2]  # Read z(t)

    #Read the Jacobian for the periodic orbit:
    J = sspJacobianSolution[-1, 3:].reshape((3, 3))

    return J

def plotFig(orbit):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(orbit[:,0], orbit[:,1], orbit[:,2])
    plt.show()


if __name__ == "__main__":
    
    case = 4
   
    # case 1: try a random initial condition
    if case == 1:
        x0 = rand(3)
        dt = 0.005
        nstp = 50.0/dt
        orbit = integrator(x0, dt, nstp)
        reduced_orbit = reduceSymmetry(orbit)
    
        plotFig(orbit)
        plotFig(reduced_orbit)

    # case 2: periodic orbit
    if case == 2:
        x0 = np.array([ -0.78844208,  -1.84888176,  18.75036186])
        dt = 0.0050279107820829149
        nstp = 156
        orbit_doulbe = integrator(x0, dt, nstp*2)
        orbit = orbit_doulbe[:nstp, :] # one prime period
        reduced_orbit = reduceSymmetry(orbit)

        plotFig(orbit_doulbe)
        plotFig(reduced_orbit)

    # case 3 : calculate Floquet multipliers and Floquet vectors associated
    # with the full periodic orbit in the full state space.
    # Please check that one of Floquet vectors is in the same/opposite
    # direction with velocity field at x0.
    if case == 3:    
        x0 = np.array([ -0.78844208,  -1.84888176,  18.75036186])
        dt = 0.0050279107820829149 # integration time step
        nstp = 156 # number of integration steps => T = nstp * dt
        
        # please fill out the part to calculate Floquet multipliers and
        # vectors.

        J = Jacobian(x0, dt, 2*nstp)
        eigenValues, eigenVectors = np.linalg.eig(J)
        print eigenValues

        v1 = np.real(eigenVectors[:,0])
        v2 = np.real(eigenVectors[:,1])
        v3 = np.real(eigenVectors[:,2])

        print v1
        print v2
        print v3

        
    # case 4: calculate Floquet multipliers and Flqouet vectors associated
    # with the prime period. 
    if case == 4:
        x0 = np.array([ -0.78844208,  -1.84888176,  18.75036186])
        dt = 0.0050279107820829149
        nstp = 156
        
        # please fill out the part to calculate Floquet multipliers and
        # vectors.

        J = Jacobian(x0, dt, nstp)
        eigenValues, eigenVectors = np.linalg.eig(J)
        print eigenValues

        v1 = np.real(eigenVectors[:0])
        v2 = np.real(eigenVectors[:1])
        v3 = np.real(eigenVectors[:2])

        print v1
        print v2
        print v3

        



        
    
    
