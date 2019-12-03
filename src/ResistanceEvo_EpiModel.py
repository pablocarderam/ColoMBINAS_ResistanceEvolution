#!/usr/bin/env python3

'''
Deterministic numerical solver for ODE systems
Pablo Cardenas R.
'''


### Imports ###
import seaborn as sns # for plots
import matplotlib.pyplot as plt

sns.set_style("white") # make pwetty plots
cb_palette = ["#999999", "#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    # http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/#a-colorblind-friendly-palette
    # http://jfly.iam.u-tokyo.ac.jp/color/

from scipy import integrate # numerical integration
import numpy as np # handle arrays
import pandas as pd # data wrangling
import joblib as jl

### Methods ###
# User-defined methods #
def initCond():

    '''
    Return initial conditions values for the model in a dictionary.
    '''

    y0 = [
        # Initial concentrations in [M] (order of state variables matters)
        1e1,        # hosts - Initial WT
        1e1         # hosts - Initial resistants
        ]

    return y0


def params():

    """
    Returns default values constant values for the model in a dictionary.
    """

    c = 1e0 # constant tunes duration of infection

    params = {
        't_0':0,        # h - Initial time value
        't_f':5,      # h - Final time value
        't_den':1e-4,    # h - Size of time step to evaluate with

        'alp':7e0,       # 1/t - treatment rate
        'bet':1e-2,     # 1/(t*hosts) - transmission intensity
        'det':1e-1,      # 1/t - recovery rate
        'eps':1,      # nondim - fitness advantage wt/res
        'N':1e3        # hosts - Host population

        # 'alp':1e0,       # 1/t - treatment rate
        # 'bet':2e-2,     # 1/(t*hosts) - transmission intensity
        # 'det':5e-1,      # 1/t - recovery rate
        # 'eps':1,      # nondim - fitness advantage wt/res
        # 'N':1e2        # hosts - Host population
    }

    return params

def solveModel():

    '''
    Main method containing all solver and plotter calls.
    Writes figures to file.
    '''

    # Set up model conditions
    p = params() # get parameter values, store in dictionary p
    y_0 = initCond() # get initial conditions
    t_vec = np.linspace(p['t_0'],p['t_f'],(p['t_f']-p['t_0'])/p['t_den'] + 1)
        # time vector based on minimum, maximum, and time step values

    # Solve model
    sol = odeSolver(odeFun,t_vec,y_0,p,solver="LSODA");

    # Call plotting of figure 1
    figTSeries(sol)

    plt.close()


def odeFun(t,y,**kwargs):

    """
    Contains system of differential equations.

    Arguments:
        t        : current time variable value
        y        : current state variable values (order matters)
        **kwargs : constant parameter values, interpolanting functions, etc.
    Returns:
        Dictionary containing dY/dt for the given state and parameter values
    """

    IW,IR = y # unpack state vars

    # Unpack variables passed through kwargs (see I thought I could avoid this
    # and I just made it messier)
    alp,bet,det,eps,N = kwargs['alp'],kwargs['bet'],kwargs['det'],kwargs['eps'], \
        kwargs['N']

    # ODEs
    dIW = ( bet * ( N + ( eps - 1 ) * IR - IW ) - det - alp ) * IW
    dIR = ( bet * ( N - ( eps + 1 ) * IW - IR ) - det ) * IR

    # Gather differential values in list (state variable order matters for
    # numerical solver)
    dy = [dIW,dIR]

    return dy


def figTSeries(sol):

    """
    This function makes a plot for Figure 1 by taking all the solution objects
    as arguments, and prints out the plot to a file.
    Arguments:
        sol : solution object taken from solver output
    """

    t_vec = sol.t[:] # get time values
    N = params()['N'] # get total pop

    plt.figure(figsize=(6, 4), dpi=200) # make new figure

    ax = plt.subplot(1, 1, 1) # Fig A
    plt.plot(t_vec, sol.y[0,:], label=r'$I_W$', color=cb_palette[3])
    plt.plot(t_vec, sol.y[1,:], label=r'$I_R$', color=cb_palette[6])
    plt.plot(t_vec, N-sol.y[0,:]-sol.y[1,:], label=r'$S$', color=cb_palette[5])
    plt.xlabel('Time')
    plt.ylabel('Hosts')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)
    #ax.set_title('Fig. 1A', loc='left')

    plt.savefig('Fig1.png', bbox_inches='tight')

# Pre-defined methods #
# These shouldn't have to be modified for different models
def odeSolver(func,t,y0,p,solver='LSODA',rtol=1e-6,atol=1e-6,**kwargs):

    """
    Numerically solves ODE system.

    Arguments:
        func     : function with system ODEs
        t        : array with time span over which to solve
        y0       : array with initial state variables
        p        : dictionary with system constant values
        solver   : algorithm used for numerical integration of system ('LSODA'
                   is a good default, use 'Radau' for very stiff problems)
        rtol     : relative tolerance of solver (1e-8)
        atol     : absolute tolerance of solver (1e-8)
        **kwargs : additional parameters to be used by ODE function (i.e.,
                   interpolation)
    Outputs:
        y : array with state value variables for every element in t
    """

    # default settings for the solver
    options = { 'RelTol':10.**-8,'AbsTol':10.**-8 }

    # takes any keyword arguments, and updates options
    options.update(kwargs)

    # runs scipy's new ode solver
    y = integrate.solve_ivp(
            lambda t_var,y: func(t_var,y,**p,**kwargs), # use a lambda function
                # to sub in all parameters using the ** double indexing operator
                # for dictionaries, pass any additional arguments through
                # **kwargs as well
            [t[0],t[-1]], # initial and final time values
            y0, # initial conditions
            method=solver, # solver method
            t_eval=t, # time point vector at which to evaluate
            rtol=rtol, # relative tolerance value
            atol=atol # absolute tolerance value
        )

    return y


# Run all if code is called as a script
if __name__ == '__main__':

    """
    Main

    """

    solveModel()

    print('\n*** Done :) ***\n')
