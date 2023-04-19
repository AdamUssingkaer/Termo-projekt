#!/usr/bin/env python
# coding: utf-8

# In[1]:


###############
# DTU 10034 Termodynamik og statistisk fysik 2023 - Projekt - Ising-modellen i 1D, 2D og 3D
# 
# Simulates the ferromagnetic Ising model without external field on a two-dimensional
# square lattice using a Metropolis-Hastings Monte Carlo algorithm.
# 
# Here, the simulation is performed for a single temperature, in order to study
# convergence. A sweep over multiple temperatures is implemented in a separate code file.
# 
# Note that energy is measured in units of the interaction energy, i.e. we set J=1
# and the sweep is understood as running over values of kB T in units of J.
###############


###############
#Import packages
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from matplotlib.pyplot import figure
from datetime import datetime
import os as os

###############
#Define the size of the lattice
nx,ny = 20,20

#Define the temperature 
T = 2.1

#Define number of steps to take in the Monte Carlo simulation. 
#the first prefactor measures how the number of flip attempts per spin
steps =1000*nx*ny

#For holding the energy
E = np.array([])
#For holding the total spin
M = np.array([])
#For holding states
S = []

#Initialize the lattice
# spin_lattice = np.ones(nx,ny) #all spins up
spin_lattice = np.random.choice([-1,1],size = (nx,ny)) #random spins
initial_spin_lattice = np.copy(spin_lattice) #store the initial state

#How often we store the energy, spin, and the state (we don't do it at every step to save time and space)
kEM = 100;
kS = 400;
 
###############
#Run the Metropolis-Hastings simulation
for k in range(steps):
           
    #Pick random site to flip
    i = np.random.randint(nx)
    j = np.random.randint(ny)
        
    #Calculate the energy change (periodic boundary condition)
    deltaE = 2*spin_lattice[i,j]*(spin_lattice[(i+1)%nx,j] + spin_lattice[i,(j+1)%ny]
                                  + spin_lattice[(i-1),j] + spin_lattice[i,(j-1)] )
    
    #Accept higher energy via Boltzmann distribution
    if np.random.random() < np.exp(-deltaE/T):
        spin_lattice[i,j] *= -1 #flip the spin
    
    #Every kEM steps, calculate and store the total energy and total spin (periodic boundary condition)
    if np.mod(k,kEM)==0:
        #Energy
        E = np.append(E, -np.sum(spin_lattice*( np.roll(spin_lattice,1,axis=0) + np.roll(spin_lattice,1,axis=1) )) )
        #Spin
        M = np.append(M,np.sum(spin_lattice))
        
        #Display progress through k loop
        print(str(int(k/steps*100))+'% of simulation done', end='\r')
        

    #Every kS steps, store the state
    if np.mod(k,kS)==0:
        S.append(np.copy(spin_lattice))
        
###############

# Print results to terminal 
# print(Tvec); print(M);  print(E) ; print(np.column_stack([Tvec,M,E])) 


# In[2]:


# Formatting plots
plt.rcParams['text.usetex'] = True # use latex formatting on labels
plt.style.use("default")
figdim = np.multiply((12,9),1/1.5) # default figure size 9 x 12 cm
px = 1/plt.rcParams['figure.dpi']  # pixel in inches


# In[3]:


# Plot state at given step and save the figure
# ks = 0; #initial
ks = len(S)-1 #final
plt.figure(1,figsize=figdim)
plt.imshow(S[ks]);
# plt.axis('off');
plt.ylabel('$x$');plt.xlabel('$y$');
plt.xticks(np.arange(0,ny,ny/10))
plt.yticks(np.arange(0,nx,nx/10))
plt.suptitle('State at step '+str(ks*kS));
plt.savefig('mmcIsing2D_state_Sim1.pdf')


# In[4]:


# Plot energy per spin vs. step
plt.figure(2,figsize=figdim)
plt.plot(np.arange(E.size)*kEM,E/(nx*ny),'o',alpha=0.5); #plot the energy per spin
plt.plot(np.array([0,steps]),np.array([0,0]),'k--',alpha=0.5) #plot line at 0 for reference
plt.xlabel('step no.');
plt.ylabel('$\\langle \\varepsilon \\rangle / J $')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.suptitle('Energy per spin')
plt.tight_layout()
plt.savefig('mmcIsing2D_EvsStep_Sim1.pdf')  


# In[5]:


# Plot magnetisation vs. step
plt.figure(3,figsize=figdim)
plt.plot(np.arange(M.size)*kEM,M/(nx*ny),'o',alpha=0.5); #plot the magnetisation
plt.plot(np.array([0,steps]),np.array([0,0]),'k--',alpha=0.5) #plot line at 0 for reference
plt.xlabel('step no.');
plt.ylabel('$\\langle m \\rangle$')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.suptitle('Magnetisation')
plt.tight_layout()
plt.savefig('mmcIsing2D_MvsStep_Sim1.pdf')  


# In[6]:


# For saving the data - useful for comparing runs with different parameter settings etc.

# # Pick a location
path = 'output/'
if not os.path.isdir(path): os.makedirs(path) #if path does not already exist then create it

# # We create a label for the numerical experiment
label = '2DIsing_singleT_Test3 '# label by date and time
label = label.replace(':',''); label = label.replace(' ','') # remove annoying chars

# # Save input parameters and output (step nos., energy, magnetization) to txt file
np.savetxt(path+label+'-input.txt', np.column_stack([nx,ny,steps,T]), fmt='%.2e')
np.savetxt(path+label+'-output.txt', np.column_stack([np.arange(E.size)*kEM,E/(nx*ny),M/(nx*ny)]), fmt='%.2e')


# In[12]:


#########################################################################################################
# Basic script to load and compare the outcome of different numerical experiments
#########################################################################################################
#Import packages
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from copy import copy
from matplotlib.pyplot import figure
from datetime import datetime

# Formatting plots
plt.rcParams['text.usetex'] = True # use latex formatting on labels
plt.style.use("default")
figdim = np.multiply((12,9),1/2.54) # default figure size 9 x 12 cm
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
path = 'output/'

# Load and plot 3 different numerical experiments

datIn = np.loadtxt(path+'2DIsing_singleT_Test1-input.txt')
datOut = np.loadtxt(path+'2DIsing_singleT_Test1-output.txt')
T0 = datIn[3];
ksteps = datOut[:,0]
E = datOut[:,2]
plt.figure(5,figsize=figdim)
plt.plot(ksteps,E,'c.',alpha=0.5);

datIn = np.loadtxt(path+'2DIsing_singleT_Test2-input.txt')
datOut = np.loadtxt(path+'2DIsing_singleT_Test2-output.txt')
T1 = datIn[3];
ksteps = datOut[:,0]
E = datOut[:,2]
plt.figure(5,figsize=figdim)
plt.plot(ksteps,E,'m.',alpha=0.5);

datIn = np.loadtxt(path+'2DIsing_singleT_Test3-input.txt')
datOut = np.loadtxt(path+'2DIsing_singleT_Test3-output.txt')
T2 = datIn[3];
ksteps = datOut[:,0]
E = datOut[:,2]
plt.figure(5,figsize=figdim)
plt.plot(ksteps,E,'y.',alpha=0.5);

# Labels
plt.xlabel('step no.');
plt.ylabel('$\\langle \\varepsilon \\rangle$')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.suptitle('Energy pr. spin')

# Create legend 
plt.legend(['$k_b T / J = '+str(T0)+'$','$k_b T / J = '+str(T1)+'$','$k_b T / J = '+str(T2)+'$'])


# In[ ]:




