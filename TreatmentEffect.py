''' ---------------------------------------------------------------------------

    Computational Econometrics
    
    FINAL EXAM 2013:    			Bocar Ba 
        
    This file is part of the Generalized Roy Toolbox. 
    This module computes the ATE, ATT, and ATU using the results from the grmEstimation.
	I used two different approaches to get the different treatment effects:
	Method 1: Based on the results from the estimation, the different treatment effects 
			are computed using a control function approach (Inverse Mill Ratio).
	
	Method 2: Based on the results from the estimation, the different treatment effects 
			are computed by simulating the distribution of the unobservales.
			
	This module contains the following:
			1)load the standard libraries
			2)Create Functions used in Method 1:
				-Inverse Mills Ratio
				-Treatment Effects
			3)Implement the 2 methods using parrallel execution
				-Method 1: Using Control Function
				-Method 2: Using Simulation
				
How to make it work?				
#module load mpi4py
#python TreatmentEffect.py
'''


'''---------------------------------------------------------------------------------------		
Load the standard libraries---------------------------------------------------------------
---------------------------------------------------------------------------------------'''
# standard library
import os
import sys
import numpy as np
import math
import json
from scipy.stats import norm

# project library
import grmToolbox
import grmReader
from mpi4py import MPI
import numpy

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()


'''---------------------------------------------------------------------------------------		
Functions-------------------------------------------------------------------------
---------------------------------------------------------------------------------------'''

'''Create functions that compute the Inverse Mills Ratio'''
def imr(muD,V_var,i):
	#Compute the Inverse Mills Ratio for the TT and TU
	if i=='ATT':
		return norm.pdf(muD/math.sqrt(V_var))/norm.cdf(muD/math.sqrt(V_var))
	elif i=='ATU':
		return norm.pdf(muD/math.sqrt(V_var))/(1-norm.cdf(muD/math.sqrt(V_var)))
	else:
		return 0
		
'''Treatment Effects (ATE, ATT, ATU) using Control Functions'''
def treatment(mu0,mu1,muD,V_var,cov1,cov0,i):
	if i=='ATE':
	#Return the ATE
		return mu1-mu0
	elif i=='ATT':
	#Return the ATT
		return mu1-mu0-((cov1-cov0)/math.sqrt(V_var))*imr(muD,V_var,'ATT')
	else:
	#Return the ATU
		return mu1-mu0+((cov1-cov0)/math.sqrt(V_var))*imr(muD,V_var,'ATU')


'''---------------------------------------------------------------------------------------		
Parallel execution to get the treatment effects-------------------------------------------
---------------------------------------------------------------------------------------'''

'''Method 1: Using Control Function....................................................'''
if rank==0:
	'''Simulation and Estimation from grmToolbox'''
	#Simulation
	grmToolbox.simulate()
	#Estimation
	grmToolbox.estimate()	

if rank==1:
	ATE_ctrl=None
	ATT_ctrl=None
	ATU_ctrl=None	
	
else:
	'Import Data'
	numCovarsOut=3
	numCovarsCost=2
	data = np.genfromtxt('grmData.dat',dtype='float')
	Y = data[:,0]
	D = data[:,1]
	X = data[:,2:(numCovarsOut + 2)]
	Z = data[:,-numCovarsCost:]

	'Import Results from Estimation'
	json_file = open('grmRslt.json')
	rslt = json.load(json_file)

	Y1_beta =rslt['Y1_beta'] 
	Y0_beta =rslt['Y0_beta']
	D_gamma =rslt['D_gamma']
	U1_var  =rslt['U1_var']
	U0_var  =rslt['U0_var']
	U1V_rho =rslt['U1V_rho']
	U0V_rho =rslt['U0V_rho'] 
	V_var   =rslt['V_var']

	#Index for D=0 and D=1
	indx0=np.where(D==0)[0]
	indx1=np.where(D==1)[0]

	#Create Y0,Y1,Zgamma
	Y0=np.dot(X[indx0,:],Y0_beta)
	Y1=np.dot(X[indx1,:],Y1_beta)
	Zgamma=np.dot(Z,D_gamma)

	#Create mu1,mu0,muD
	mu0=Y0.mean(axis=0)
	mu1=Y1.mean(axis=0)
	muD=Zgamma.mean(axis=0)

	#Compute the covariances
	cov1 =U1V_rho*math.sqrt(V_var)*math.sqrt(U1_var)
	cov0 =U0V_rho*math.sqrt(V_var)*math.sqrt(U0_var)
	
	''' Treatment Effects'''
	ATE_ctrl=treatment(mu0,mu1,muD,V_var,cov1,cov0,'ATE')
	ATT_ctrl=treatment(mu0,mu1,muD,V_var,cov1,cov0,'ATT')
	ATU_ctrl=treatment(mu0,mu1,muD,V_var,cov1,cov0,'ATU')
	
print "Treatment Effects Using the Control Function Approach"
print "the ATE is", ATE_ctrl
print "the ATT is", ATT_ctrl
print "the ATU is", ATU_ctrl


'''Method 2: Using Simulation..........................................................'''
if rank==1:
	ATE_sim=None
	ATT_sim=None
	ATU_sim=None	
		
else:
	'Import Data'
	numCovarsOut=3
	numCovarsCost=2
	data = np.genfromtxt('grmData.dat',dtype='float')
	Y = data[:,0]
	D = data[:,1]
	X = data[:,2:(numCovarsOut + 2)]
	Z = data[:,-numCovarsCost:]

	'Import Results from Estimation'
	json_file = open('grmRslt.json')
	rslt = json.load(json_file)

	Y1_beta =rslt['Y1_beta'] 
	Y0_beta =rslt['Y0_beta']
	D_gamma =rslt['D_gamma']
	U1_var  =rslt['U1_var']
	U0_var  =rslt['U0_var']
	U1V_rho =rslt['U1V_rho']
	U0V_rho =rslt['U0V_rho'] 
	V_var   =rslt['V_var']
	rndSeed =rslt['randomSeed']
	N 		=rslt['numAgents']
	
	#Create Y0_level,Y1_level,D_level
	Y1_level = np.dot(Y1_beta, X.T)
	Y0_level = np.dot(Y0_beta, X.T)
	D_level  = np.dot(D_gamma, Z.T)

	#Covariances
	cov1 =U1V_rho*math.sqrt(V_var)*math.sqrt(U1_var)
	cov0 =U0V_rho*math.sqrt(V_var)*math.sqrt(U0_var)

	#Simulate unobservables from the model.
	means = np.tile(0.0, 3)
	vars_ = [U1_var, U0_var, V_var]
	covs  = np.diag(vars_)
	covs[0,2] = cov1 
	covs[2,0] = covs[0,2]
	covs[1,2] = cov0
	covs[2,1] = covs[1,2]
	U = np.random.multivariate_normal(means, covs, N)
	
	'''Simulate individual outcomes and choices'''
	# Unobservables
	U1 = U[:,0]
	U0 = U[:,1]
	V  = U[:,2]

	# Potential outcomes.
	Y1 = Y1_level + U1
	Y0 = Y0_level + U0
    
	# Some calculations outside the loop
	EB = Y1_level - Y0_level

	# Decision Rule.
	cost = D_level  + V     
	D = np.array((EB - cost > 0))
        
	# Observed outcomes.
	Y  = D*Y1 + (1.0 - D)*Y0
	
	''' Check quality of simulated sample / anti-bugging'''
	assert (np.all(np.isfinite(Y1)))
	assert (np.all(np.isfinite(Y0)))
	assert (np.all(np.isfinite(Y)))
	assert (np.all(np.isfinite(D)))
	assert (Y1.shape == (N, ))
	assert (Y0.shape == (N, ))    
	assert (Y.shape  == (N, ))
	assert (D.shape  == (N, ))    
	assert (Y1.dtype == 'float')
	assert (Y0.dtype == 'float')    
	assert (Y.dtype == 'float')    
	assert ((D.all() in [1.0, 0.0]))

	''' Treatment Effects'''
	ATE_sim = (Y1-Y0).sum() / N
	ATT_sim =  ((Y1-Y0)*D).sum() / np.sum(D)
	ATU_sim =  np.sum((Y1-Y0)*(1-D)) / np.sum(1-D)
	
print "Treatment Effects Using the Simulation Approach"
print "the ATE is", ATE_sim
print "the ATT is", ATT_sim
print "the ATU is", ATU_sim