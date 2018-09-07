import numpy as np,cPickle

class line_candidate(object):
	def dist3d(self,pos1,pos2):
		'''
		Calculate the 3D distance of a pair of triplets in the datacube.
		'''
		return np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2+(pos1[2]-pos2[2])**2)
	def dist2d(self,pos1,pos2):
		'''
		Calculate the 2D spatial distance of a pair of triplets in the datacube.
		'''
		return np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)
	def __init__(self,reduint_entry):
		'''
		Initialize the object by specifying the standard line candidate tuple.

		Parameters
		reduint_entry: tuple
			Line candidate basic properties (SNR,(position),Ntempl, (peak_template))

		'''
		self.SNR=reduint_entry[0]
		self.posn=reduint_entry[1]
		if self.dist2d(self.posn,(257, 345,0))<5:
			print 'this is probably continuum'
		self.spat_templ=reduint_entry[3][0]
		self.freq_templ=reduint_entry[3][1]
		self.reduint_entry=reduint_entry
		#Finds the measured aperture-extracted properties.
		inp=open('/data2/common/COdeep_cosmos/MFanalysis/spectra_nosmooth_pos.txt')
		for idx,line in enumerate(inp):
			splitt=line.split()
			if float(splitt[0][1:-1])==self.SNR:
				try:
					self.aper_flux=float(splitt[13])
					self.aper_maj=float(splitt[7])
					self.aper_min=float(splitt[9])
					self.aper_freq=float(splitt[11])
					self.aper_FWHM=float(splitt[15])
					self.aper_int_flux=float(splitt[17])
				except:
					print 'Aper flux reading failed'
					self.aper_flux=np.nan
					self.aper_maj=np.nan
					self.aper_min=np.nan
					self.aper_freq=np.nan
					self.aper_FWHM=np.nan
					self.aper_int_flux=np.nan
				break
		inp.close()
		#Finds the measured single-pixel-extracted properties.
		inp=open('/data2/common/COdeep_cosmos/MFanalysis/spectra_nosmooth_1pix_pos.txt')
		for idx,line in enumerate(inp):
			splitt=line.split()
			if float(splitt[0][1:-1])==self.SNR:
				try:
					self.pix_flux=float(splitt[9])
					self.pix_freq=float(splitt[7])
					self.pix_FWHM=float(splitt[11])
					self.pix_int_flux=float(splitt[13])
				except:
					print '1pix flux reading failed'
					self.pix_flux=np.nan
					self.pix_freq=np.nan
					self.pix_FWHM=np.nan
					self.pix_int_flux=np.nan
				break
		inp.close()
		#Assigns the appropriate candidate purity
		if self.SNR>6:
			self.purity=1
		else:
			self.assign_purity()
		#Assigns the appropriate flux-factor distribution function, utilizing the size probability distribution.
		self.assign_flux_corr()
		inp=open('/data2/common/COdeep_cosmos/artificial/no_smooth/post_size_881002.dat')
		posterior=cPickle.load(inp)
		inp.close()
		SNRbins=np.arange(4,7,.1)
		SNRbin_idx=np.digitize(self.SNR,SNRbins)-1
		self.size_prob=np.array([np.mean(posterior[self.spat_templ][inj_size][(SNRbin_idx-3):(SNRbin_idx+3)]) for inj_size in range(3)])
		inp=open('/data2/common/COdeep_cosmos/artificial/no_smooth/post_FWHM_343333.dat')
		posterior_freq=cPickle.load(inp)
		inp.close()
		self.FWHM_prob=np.array([np.mean(posterior_freq[self.freq_templ][inj_FWHM][(SNRbin_idx-3):(SNRbin_idx+3)]) for inj_FWHM in range(3)])
		#Assigns the candidate completeness
		self.completeness=self.assign_compl(self.aper_int_flux)
		#Calculates the line luminosity
		self.L_prime=self.calc_L_prime(self.aper_int_flux,self.aper_freq)
		#Calculates the RA,Dec coordinates
		self.ra,self.dec=self.calc_ra_dec_freq(self.posn)[:2]
	def assign_purity(self):
		'''
		Finds the appropriate purity from the saved list of computed purities by matching the SNR exactly.
		'''
		purity_list=np.load('purity_list.npy')
		temp_pur=[pur[1] for pur in purity_list if pur[0]==self.SNR] 
		if len(temp_pur)>0:
			self.purity=temp_pur[0]
		else:
			self.purity=0
	def assign_flux_corr(self):
		'''
		Determines the appropriate distribution function for the flux-factor correction, based on the prviously fit lognormal models.
		'''
		if self.SNR>6:
			self.flux_correct=1.
			self.flux_correct_range=(1.,1.)
			from scipy.stats import norm
			self.bestf=norm(loc=1.,scale=1e-8)
		else:
			lognorm_param={-1:[ 0.00618493  ,0.39608951],0:[ 0.32328394,  0.53686663],2:[ 0.17125074 , 0.54813403],4:[ 0.18224569  ,0.51624523],6:[ 0.29691822 , 0.56654581],8:[ 0.42054354,  0.61816686]}
			from scipy.stats import lognorm
			self.bestf=lognorm(s=lognorm_param[min(8,self.spat_templ)][1],scale=np.exp(lognorm_param[min(8,self.spat_templ)][0])) 
			self.flux_correct=self.bestf.median()
			self.flux_correct_range=self.bestf.interval(.68)
	def assign_compl(self,flux_to_use):
		'''
		Calculates the completeness for the line candidate given the flux and the probability distribution for line size and frequency width.

		Parameters
		flux_to_use: float
			The Integrated line flux to use to evaluate the completeness.
		'''
		if np.isnan(flux_to_use):
			return np.nan
		myfit=lambda f,d,f0: max(0,1-(   1./(f+d)*np.exp(-f/f0)   ))
		compl_params_fit=np.load('completeness_params.npy') 
		comple=0
		for spat_bin in [0,1,2]:
			for freq_bin in [0,1,2]:
				param=compl_params_fit[spat_bin,freq_bin]
				comple+=self.size_prob[spat_bin]*self.FWHM_prob[freq_bin]*myfit(flux_to_use,*param)
		return max(.1,comple)
	def calc_L_prime(self,sdv,nu_obs,J=1):
		'''
		Applies the standard formula to compute line L' luminosity from the line flux and redshift.
		'''
		z=115.27*J/nu_obs-1
		from astropy.cosmology import FlatLambdaCDM
		cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
		DL=cosmo.luminosity_distance(z).value
		return 3.25e7*sdv*DL**2/(1+z)**3/nu_obs**2      
	def completeness_withbins(self,complet_bins):
		'''
		Computes the completeness by weighing the input matrix by the spatial size and frequency distributions for the line candidate.
		'''
		return np.sum(np.outer(self.size_prob,self.FWHM_prob).flatten()*np.array(complet_bins))
	def calc_ra_dec_freq(self,posn):
		'''
		Computes the absolute coordinates given a position within the cube.

		Parameters

		posn: tuple of float or int
			Coordinates within the data cube

		Returns
		
		Ndarray with the RA,Dec, and frequency in Hz
		'''
		import pyfits,pywcs
		f=pyfits.open('/data2/common/COdeep_cosmos/singlepointings/NewCOSMOS20.fits')
		mywcs=pywcs.WCS(f[0].header)
		return mywcs.wcs_pix2sky(np.array([posn]),0)[0][:3]


