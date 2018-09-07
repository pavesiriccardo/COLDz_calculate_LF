import numpy as np,cPickle,os
from mpmath import *
from scipy.stats import gamma,poisson
from scipy import special as sp
from multiprocessing import Pool


import cand_class   #This is my "line candidate class"

mp.dps = 8; mp.pretty = True


def simulate_schechter_distribution( alpha, L_star, L_min, N):
		'''
		This draws samples from a Schechter distribution

		Parameters:

		alpha: float
			This faint slope index

		L_star: float
			The Schechter knee parameter

		L_min: float
			The low L' cutoff

		N: int
			The number of samples to draw

		Returns:

		Ndarray of L' line luminosities in K km/s pc^2 drawn from the distribution

		'''
		import numpy as np
		if N==0:
			return np.array([])
		n=0
		output = []
		while n<N:
			if alpha<-.8:
				L = np.random.gamma(scale=L_star, shape=alpha+1, size=10*N) #50
			else:
				L = np.random.gamma(scale=L_star, shape=alpha+1, size=N*2)
			L = L[L>L_min]
			u = np.random.uniform(size=L.size)
			L = L[u<L_min/L]
			output.append(L)
			n+=L.size
		return np.concatenate(output)[:N]




def calc_L_prime(sdv,nu_obs):
		'''
		This return the L' line luminosity 

		Parameters:

		sdv:float
			The Integrated line flux in Jy

		nu_obs: float
			The observed line frequency in GHz

		Return:
			The L' line luminosity in K km/s pc^2

		'''
		z=115.27/nu_obs-1
		from astropy.cosmology import FlatLambdaCDM
		cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
		DL=cosmo.luminosity_distance(z).value
		return 3.25e7*sdv*DL**2/(1+z)**3/nu_obs**2   





#Load the statistical correction factors I calculated elsewhere

likelihood=list(np.load('spatial_likelihood.npy'))
likelihood_freq=list(np.load('frequency_likelihood.npy'))
lognorm_params_given_inj_and_meas=np.load('flux_factor_params.npy')
L_prime_freq_conversion=np.load('/data2/common/goodsN/SNRanalysis_57w/MF3D/L_prime_freq_conversion.npy')
compl_params_fit=np.load('completeness_params.npy')   


#Load the list of line candidates
#Each object in the list is a standard tuple of form: (SNR,(position), Ntempl, (Peak_template))

inp=open('../reduint_nosmooth_combined_pos.dat')
reduint=cPickle.load(inp)
inp.close()

#Remove known duplicates
toremove=[0,4,5,9]
reduint=[red for idx,red in enumerate(reduint) if idx not in toremove]

#Initialize objects for each of the line candidates
objects=[cand_class.line_candidate(red) for red in reduint if red[0]>5.25] 

########################



def check_one_value(params): 
	'''
	This function checks a value of the Schechter distribution parameters, 
	and returns their value if they produce a set of "galaxies" with line fluxes close to the observed ones.
	
	Parameters:
	params: tuple of float
		logLstar,logPhiStar,alpha are the standard Schechter parameters
	'''
	#The Schechter distribution parameters
	logLstar,logPhiStar,alpha=params   
	logLmin=8.5 
	#Cosmic volume sampled in Mpc^3:
	vol=20189.  
	#This choice of alpha means that the LF form is NOT same as Mike's but rather has alpha in the exponent when written as dN/dV/dlogL
	#This is how many galaxies in volume vol, are expected on average:
	Numexp=10**(logPhiStar)*vol*fp.gammainc(alpha,10**(logLmin-logLstar))   
	#Draw galaxies from a Poisson distribution:
	N_to_use=poisson.rvs(Numexp)   
	#Draw their luminosities from the appropriate Schechter function:
	Lprimes=simulate_schechter_distribution( alpha, 10**logLstar, 10**logLmin, N_to_use)   
	#Assign random line frequencies to each:
	freq_list=31+8*np.random.random(N_to_use)    
	#Calculate the observed flux from the intrinsic luminosity and the distance:
	sdv_list=Lprimes/L_prime_freq_conversion[((freq_list-30)/9.0e-5).astype(int)]   
	#Randomly choose a spatial size for each
	spat_real=np.random.choice(3, size=N_to_use, replace=True, p=[0.88, 0.1, 0.02])   
	#Randomly choose a frequency width for each
	freq_real=np.random.choice(3, size=N_to_use, replace=True, p=[0.34, 0.33, 0.33])  
	#Define the completeness fitting function 
	myfit=lambda f,d,f0: max(0,1-(   1./(f+d)*np.exp(-f/f0)   ))   
	#Calculate the completeness for each of the simulated "galaxies"
	completenesses=[myfit(sdv,*compl_params_fit[spat,freq]) for sdv,spat,freq in zip(sdv_list,spat_real,freq_real)]  
	#Consider them observed with the correct probability
	observed=np.random.random(size=N_to_use)<completenesses 
	#How many are expected to be observed
	Nobs=np.sum(observed)  
	#Now take care of the real candidates
	#This implements "Normal" purity
	#purities=np.array([max(0,obj.purity*np.random.normal(loc=1.,scale=1.))  if obj.purity<1 else 1 for obj in objects]) 
	#This implements "Uniform" purity
	purities=np.array([np.random.random()*obj.purity if obj.purity<1 else 1 for obj in objects ])   
	#Draw the real galaxies from the observed ones with appropriate probability.
	selected_candidates=np.random.random(size=len(objects))<purities      
	Nselected=np.sum(selected_candidates)
	if Nselected==Nobs:   
		Lprime_observed_real=np.array([obj.L_prime  for  idx,obj in enumerate(objects)  if selected_candidates[idx]  ])
		SNRbins=np.arange(4,7,.1)
		SNRbin_idx=np.digitize(5.5,SNRbins)-1   
		observed_Lprime=[]
		#Loop over each "simulated" galaxy candidate, applying the flux corrections appropriate for its size, to predict the observed line luminosity
		for idx in range(Nobs):
			inj_spa=spat_real[observed][idx]
			inj_fre=freq_real[observed][idx]
			spat_distrib=[np.mean([likelihood[inj_spa][key][SNR_b]  for SNR_b in range(SNRbin_idx-3,SNRbin_idx+3)]) for  key in [-1,0,2,4,6,8,10,12]]
			spat_obs_idx=min(5,np.random.choice(8, size=1, replace=True, p=spat_distrib)[0])
			spat_obs=[-1,0,2,4,6,8][spat_obs_idx]
			freq_distrib=[np.mean([likelihood_freq[inj_fre][key][SNR_b]  for SNR_b in range(SNRbin_idx-3,SNRbin_idx+3)]) for  key in [4,8,12,16,20]]
			freq_obs=[4,8,12,16,20][np.random.choice(5, size=1, replace=True, p=freq_distrib)[0]]
			#for the given intrinsic and observed properties, and the SNR=5.5, what flux ratio do we expect?
			lognorm_par=lognorm_params_given_inj_and_meas[spat_obs_idx,inj_spa]
			from scipy.stats import lognorm
			flux_correct=lognorm(s=lognorm_par[1],scale=np.exp(lognorm_par[0])).rvs()
			observed_flux=sdv_list[observed][idx]*flux_correct			
			observed_Lprime.append(Lprimes[observed][idx]*flux_correct)
		allmatched=True
		for obs_real_Lprime in Lprime_observed_real:
			#If they all match in observed luminosity to better than 20%, then success!
			if np.min(np.absolute(obs_real_Lprime-np.array(observed_Lprime))/obs_real_Lprime)>.2:    
				allmatched=False
		if allmatched:
			return params



Nsamp=10000

p = Pool(30)

samples=[] #list(np.loadtxt('mysamples'))

#Uses a pool of 30 processes to run in parallel.
#Run until we have found at least 500 samples.
while (len(samples)<500):
	#logLstar,logPhiStar,alpha
	#Define the prior ranges:
	prior_samples=np.array(zip(np.random.random(size=Nsamp)*2+9.5,np.random.random(size=Nsamp)*2.5-5,np.random.random(size=Nsamp)*2.-.9 ))
	samples+=[x for x in  p.map(check_one_value,prior_samples) if x is not None]
	np.savetxt('mysamples',samples)


'''
samp=np.loadtxt('mysamples')
samp.shape
import corner
corner.corner(samp,labels=['L_prime','Phi_prime','alpha'])
'''



