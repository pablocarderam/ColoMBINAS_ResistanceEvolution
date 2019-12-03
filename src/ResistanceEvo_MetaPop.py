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

# Global param
# Event ID constants
Ti = 0 # Transmission event from infected
Re = 1 # Recovery event
Ta = 2 # Treatment event

# State var ID constants
t  = -1 # time (last in array, hence -1)
PW = 0 # Wild type pathogens
PR = 1 # Resistant pathogens

### Methods ###
# User-defined methods #
def initCond():

    '''
    Return initial conditions values for the model in a dictionary.
    '''

    y0 = [
        # Initial concentrations in [M] (order of state variables matters)
        3e2,        # pathogens - Initial WT
        1e3         # pathogens - Initial resistants
        ]

    return y0


def params():

    """
    Returns default values constant values for the model in a dictionary.
    """

    c = 5e1 # constant tunes duration of infection

    params = {
        't_0':0,        # h - Initial time value
        't_f':1,      # h - Final time value
        't_den':1e-2,    # h - Size of time step to evaluate with

        'rW':1*c,       # cells/t - Specific growth rate WT
        'rR':1*c,     # cells/t - Specific growth rate resistant
        'aWR':1.1,      # nondim - Effect of resistant on WT
        'aRW':2,      # nondim - Effect of WT on resistant
        'k':1e7,        # cells - Host resource carrying capacity
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

    # Unpack variables passed through kwargs (see I thought I could avoid this
    # and I just made it messier)
    rW,rR,k,aWR,aRW = kwargs['rW'],kwargs['rR'],kwargs['k'],kwargs['aWR'], \
        kwargs['aRW']

    # ODEs
    dPW = rW * ( 1 - ( ( y[PW] + aWR * y[PR] ) / k ) ) * y[PW]
    dPR = rR * ( 1 - ( ( y[PR] + aRW * y[PW] ) / k ) ) * y[PR]

    # Gather differential values in list (state variable order matters for
    # numerical solver)
    dy = [dPW,dPR]

    return dy


def figTSeries(sol):

    """
    This function makes a plot for Figure 1 by taking all the solution objects
    as arguments, and prints out the plot to a file.
    Arguments:
        sol : solution object taken from solver output
    """

    t_vec = sol.t[:] # get time values

    plt.figure(figsize=(6, 4), dpi=200) # make new figure

    ax = plt.subplot(1, 1, 1) # Fig A
    plt.plot(t_vec, sol.y[PW,:], label=r'$P_W$', color=cb_palette[3])
    plt.plot(t_vec, sol.y[PR,:], label=r'$P_R$', color=cb_palette[6])
    plt.xlabel('Time')
    plt.ylabel('Pathogens')
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


### Stochastic
class StocModel(object):

    """
    Class defines a model's parameters and methods for changing system state
    according to the possible events and simulating a timecourse using the
    Gillespie algorithm.
    """

    def __init__(self, t_vec, n_hos=100):

        '''
        Class constructor defines parameters and declares state variables.
        Arguments:
            t_vec    : s - array with times at which to evaluate simulation
        '''

        super(StocModel, self).__init__() # initialize as parent class object

        self.n_hos = n_hos # number of hosts

        # Event modifier constants
        self.alp = 0     # 1 * hosts^-1 - Frequency at which treatment is given
        self.bet = 2e-2     # 1 * hosts^-1 - Vector density-dependent transmission, accounts for only probability of transmission event, normalized to total host population
        self.gam = 1e-4     # 1/pathogens - efficiency of pathogen transmission given a transmission event; determines number of pathogens sampled from infecting host and transmitted during successfull transmission event
        self.det = 5e0     # 1 * hosts^-1 - Frequency at which recovery occurs

        self.params = params() # carrying capacity of host

        # Event IDs
        self.evt_IDs = [ Ti, Re, Ta ]
            # event IDs in specific order

        # State variables
        self.t_var = 0.0 # s - time elapsed in simulation
        self.x_var = np.zeros( [ 4,self.n_hos ] )
            # state vectors (matrix due to multiple hosts)
            # state variable order: t, PW, PR, A
        self.x_var[t,:] = 0 # start at time 0
        self.x_var[PW,:] = 0 # start at no pathogens
        self.x_var[PR,:] = 0 # start at no pathogens
        #self.inf = {} # infected compartment set

        # Array trackers
        self.t = t_vec
            # s - array with time values at which to evaluate simulation
        self.x = np.empty( [ *self.x_var.shape ] )


    def getRates(self):

        '''
        Calculates event rates according to current system state.
        Returns:
            dictionary with event ID constants as keys and rates as values.
                Includes total rate under 'tot' key.
        '''

        rates = np.zeros( [ len(self.evt_IDs) ] )
            # rate array size of event space

        HI = np.sum( ( self.x_var[PW,:] + self.x_var[PR,:] ) > 0 ) # infected hosts

        rates[Ti] = self.bet * HI
        rates[Re] = self.det * HI
        rates[Ta] = self.alp * HI

        return rates


    def doAction(self,act,hos1,hos2=None):

        '''
        Changes system state variables according to act argument passed (must be
        one of the event ID constants)
        Arguments:
            act : int event ID constant - defines action to be taken
        '''

        # Set up model conditions
        y_0 = self.x_var[PW:PR+1,hos1] # get initial conditions
        t_vec = np.linspace(self.x_var[t,hos1],self.t_var,(self.t_var-self.x_var[t,hos1])/self.params['t_den'] + 2)
            # time vector based on minimum, maximum, and time step values

        # Solve model
        self.x_var[PW:PR+1,hos1] = odeSolver(odeFun,t_vec,y_0,self.params,solver="LSODA").y[:,-1] # simulate and update state variables for host
        self.x_var[t,hos1] = self.t_var # update time for this host

        if act == Ti:
            inf_PW = np.random.poisson( max( np.floor( self.x_var[PW,hos1] * self.gam ), 0 ) ) # number of infecting WT pathogens
            inf_PR = np.random.poisson( max( np.floor( self.x_var[PR,hos1] * self.gam ), 0 ) ) # number of infecting resistant pathogens

            self.x_var[PW,hos2] += inf_PW # add infecting dose
            self.x_var[PR,hos2] += inf_PR # add infecting dose
            # print(inf_PW,inf_PR)
        elif act == Re:
            self.x_var[PW,hos1] = 0 # kill susceptible pop
            self.x_var[PR,hos1] = 0 # kill resistant pop
        elif act == Ta:
            self.x_var[PW,hos1] = 0 # kill susceptible pop


    def gillespie(self):

        '''
        Simulates a time series with time values specified in argument t_vec
        using the Gillespie algorithm. Stops simulation at maximum time or if no
        change in distance has occurred after the time specified in
        max_no_change. Records position values for the given time values in
        x_vec.
        '''

        # Simulation variables
        self.t_var = self.t[0] # keeps track of time
        self.x[:,:] = self.x_var # initialize distance at current distance
        intervention_tracker = False # keeps track of what the next intervention should be

        while self.t_var < self.t[-1]:
                # repeat until t reaches end of timecourse
            r = self.getRates() # get event rates in this state
            print(self.t_var,r)
            r_tot = np.sum(r) # sum of all rates
            if not intervention_tracker and self.t_var > -1: # if there are any interventions left and if it is time to make one, # SPECIFY INTERVENTION TIME HERE
                # self.x_var[R,0] = 0 # SPECIFY INTERVENTION ACTION HERE

                intervention_tracker = True # advance the tracker

            if 0 < r_tot: # if average time to next event is less than infinite
                # allow it, calculate probability
                # Time handling
                dt = np.random.exponential( 1/r_tot ) # time until next event
                self.t_var += dt # add time step to main timer

                # Event handling
                if self.t_var < self.t[-1]: # if still within max time
                    u = np.random.random() * r_tot
                        # random uniform number between 0 (inc) and total rate (exc)
                    r_cum = 0 # cumulative rate
                    for e in range(r.shape[0]): # for every possible event,
                        r_cum += r[e] # add this event's rate to cumulative rate
                        if u < r_cum: # if random number is under cumulative rate
                            inf = np.arange( self.n_hos )[ self.x_var[PW,:] + self.x_var[PR,:] > 0 ] # indeces of infected hosts
                            h_rand =  inf.shape[0] * ( u-r_cum+r[e] ) / r[e] # random number used to establish hosts
                            hos1 = inf[ int( np.floor( h_rand ) ) ] # first host (from list of infected)
                            hos2 = int( np.floor( self.n_hos * ( h_rand % 1) ) ) # second host (only used for transmission), based on decimal places of h_rand
                            self.doAction(e,hos1,hos2) # do corresponding action
                            break # exit event loop


                        else: # if the inner loop wasn't broken,
                            continue # continue outer loop

                        break # otherwise, break outer loop



            else: # if no more events happening,
                self.t_var = self.t[-1] # run time to the end

            self.x = np.dstack( (self.x, self.x_var) ) # record state variables in list
            self.t = self.t[0:-1] + [self.t_var,self.t[-1]] # add this time point

        self.t = self.t[0:-1] # remove last time point (used to check simulation end) to get correspoding x, t dimensions


def simModel(t_vec, n_hos, IW_0, IR_0, alp=0, bet=2e-1):

    '''
    Creates and simulates stochastic model 'a la Gillespie.
    Arguments:
        t_vec   : s - time vector for simulations
    Returns:
        StocModel object after simulation
    '''

    m = StocModel(t_vec,n_hos) # create model object
    m.alp = alp # set parameter
    m.bet = bet # set parameter

    iw = np.random.randint(0,m.n_hos,IW_0) # indeces of wt infected hosts
    ir = np.random.randint(0,m.n_hos,IR_0) # indeces of res infected hosts
    m.x_var[PW,iw] = 5e1 # initial infection conditions
    m.x_var[PR,ir] = 5e1 # initial infection conditions

    m.gillespie() # simulate

    m.com = np.sum( m.x > 0, 1) # aggregate compartments
    m.coi = np.sum( m.x[PW,:,:] * m.x[PR,:,:] > 0, 0) # aggregate compartments

    return m

def multiSim(n_iter,t_vec, n_hos, IW_0, IR_0, alp=0, bet=2e-1):

    '''
    Performs multiple simulations using embarrassing parallelization.
    Arguments:
        n_iter  : int - total number of iterations of simulation to run
        t_vec   : s - time vector for simulations
    Returns:
        list of StocModel objects after simulation
    '''

    r = jl.Parallel(n_jobs=5, verbose=10) \
        ( jl.delayed(simModel)(t_vec, n_hos, IW_0, IR_0, alp, bet) for _ in range(n_iter) )
        # parallelizes simulations across all available cores
        # verbose 10 shows progress bar

    return r

def figStochastic(sols):

    """
    This function makes a plot for Figure 2 by taking all the solution objects
    as arguments, and prints out the plot to a file.
    Arguments:
        sol : solution object taken from solver output
    """

    plt.figure(figsize=(6, 4), dpi=200) # make new figure

    ax = plt.subplot(2, 1, 1) # Fig A
    plt.plot(sols[0].t[:], sols[0].com[PW,:]/sols[0].n_hos, label=r'$I_W$', alpha=1, color=cb_palette[3])
    plt.plot(sols[0].t[:], sols[0].com[PR,:]/sols[0].n_hos, label=r'$I_R$', alpha=1, color=cb_palette[6])
    for sol in sols: # for every solution
        plt.plot(sol.t[:], sol.com[PW,:]/sol.n_hos, alpha=0.2, color=cb_palette[3])
        plt.plot(sol.t[:], sol.com[PR,:]/sol.n_hos, alpha=0.2, color=cb_palette[6])

    plt.xlabel('Time')
    plt.ylabel('Fraction of population')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)

    ax = plt.subplot(2, 1, 2) # Fig A
    plt.plot(sols[0].t[:], sols[0].coi/sols[0].n_hos, label=r'$I_{W,R}$', alpha=1, color=cb_palette[1])
    plt.plot(sols[0].t[:], (sols[0].n_hos-sols[0].com[PW,:]-sols[0].com[PR,:]+sols[0].coi)/sols[0].n_hos, label=r'$S$', alpha=1, color=cb_palette[0])
    for sol in sols: # for every solution
        plt.plot(sol.t[:], sol.coi/sol.n_hos, alpha=0.2, color=cb_palette[1])
        plt.plot(sol.t[:], (sol.n_hos-sol.com[PW,:]-sol.com[PR,:]+sol.coi)/sol.n_hos, alpha=0.2, color=cb_palette[0])

    plt.xlabel('Time')
    plt.ylabel('Fraction of population')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)

    plt.savefig('Fig2.png', bbox_inches='tight')


def figViolinPlots(sols_arr,names,filename,col):

    """
    This function makes a plot for Figure 2 by taking all the solution objects
    as arguments, and prints out the plot to a file.
    Arguments:
        sol : solution object taken from solver output
    """

    dat = pd.DataFrame(columns=['Condition','Population type','Fraction of population']) # will contain all simulation endpoint data

    for i in range( len( sols_arr ) ): # for every condition simulated
        sols = sols_arr[i] # array of replicate simulations of a given condition
        d = pd.DataFrame(
            np.transpose( np.array([
                [ names[i] ] * ( len(sols) * 4 ), # Condition name for every value in every replicate across each of the four compartments graphed
                [ r'I_W' ] * ( len(sols) ) + [ r'I_R' ] * ( len(sols) ) + [ r'I_{W,R}' ] * ( len(sols) ) + [ r'S' ] * ( len(sols) ), # Compartment names for every value
                [ s.com[PW,-1]/s.n_hos for s in sols ] + [ s.com[PR,-1]/s.n_hos for s in sols ] + [ s.coi[-1]/s.n_hos for s in sols ] +
                    [ ( s.n_hos - s.com[PW,-1] - s.com[PR,-1] + s.coi[-1] )/s.n_hos for s in sols ] # store endpoints for each compartment as fraction of population
            ]) ),
            columns=['Condition','Population type','Fraction of population']
        )

        dat = dat.append( d, ignore_index=True ) # append to existing dataframe

    #dat['Condition'] = pd.to_numeric( dat['Condition'] ) # cast as numeric
    dat['Fraction of population'] = pd.to_numeric( dat['Fraction of population'] ) # cast as numeric
    dat.to_csv(filename+'.csv', index=False) # save data

    plt.figure(figsize=(6, 4), dpi=200) # make new figure
    ax = plt.subplot(1, 1, 1) # Fig A
    ax = sns.violinplot(x="Population type", y="Fraction of population", hue="Condition", data=dat, palette=sns.cubehelix_palette(5, start=col, rot=-.75) ) # violinplots
    ax.set_xticklabels( [r'$I_W$',r'$I_R$',r'$I_{W,R}$',r'$S$'] ) # set x labels
    #ax.legend(title= 'Condition',loc='top right',labels=names)
    #ax.legend( names )
    #for t, l in zip(ax._legend.texts, names): t.set_text(l) # add legend text

    plt.savefig(filename+'.png', bbox_inches='tight')

# Run all if code is called as a script
if __name__ == '__main__':

    """
    Main

    """

    solveModel()

    # Figure 2 - meta model
    print('\n*** Starting Fig 2 Sims ***')
    # create, simulate model objects with the given time series and [ATP] in M
    t_vec = [0,5] # time vector endpoints to evaluate
    m = simModel( t_vec, 100, 30, 30, alp=5e-1, bet=1e1 )
    figStochastic([m]) # plot
    ms = multiSim(10, t_vec, 100, 30, 30, alp=0, bet=1e1)
    figStochastic(ms) # plot


    t_vec = [0,5] # time vector endpoints to evaluate
    reps = 10 # replicates

    print('\n*** Starting Fig 3 Sims ***')
    ms_a1 = multiSim(reps, t_vec, 100, 30, 30, alp=0, bet=1e1)
    ms_a2 = multiSim(reps, t_vec, 100, 30, 30, alp=5e-1, bet=1e1 )
    ms_a3 = multiSim(reps, t_vec, 100, 30, 30, alp=1e0, bet=1e1 )
    figViolinPlots([ms_a1,ms_a2,ms_a3],['alpha=0','alpha=0.5','alpha=1.0'],'Fig3',2.5)

    print('\n*** Starting Fig 4 Sims ***')
    ms_a1 = multiSim(reps, t_vec, 100, 30, 30, alp=5e-1, bet=8e0)
    ms_a3 = multiSim(reps, t_vec, 100, 30, 30, alp=5e-1, bet=15e0)
    #ms_a2 = multiSim(reps, t_vec, 100, 30, 30, alp=5e-1, bet=2e1)
    figViolinPlots([ms_a1,ms_a3,ms_a2],[r'beta=8',r'beta=8',r'beta=10'],'Fig4',0.5)

    print('\n*** Starting Fig 5 Sims ***')
    ms_a1 = multiSim(reps, t_vec, 200, 30, 30, alp=5e-1, bet=1e1)
    ms_a2 = multiSim(reps, t_vec, 100, 30, 30, alp=5e-1, bet=1e1)
    ms_a3 = multiSim(reps, t_vec, 300, 30, 30, alp=5e-1, bet=1e1)
    figViolinPlots([ms_a2,ms_a1,ms_a3],[r'N=100',r'N=200',r'N=300'],'Fig5',1)

    print('\n*** Done :) ***\n')
