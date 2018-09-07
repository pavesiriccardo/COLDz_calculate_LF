# COLDz_calculate_LF
Scripts to use the COLDz galaxy line candidates to constrain the CO Luminosity Function at z=2-3, using the ABC method.

This code was developed as part of the COLDz project to measure the CO luminosity function at high redshift. See coldz.astro.cornell.edu for more details, Pavesi et al. 2018b (1808.04372) and Riechers et al. 2018 (1808.04371) for reference. 

See the other COLDz repos for additional code related to this project.
This code uses the line candidates selected by MF3D (see repo), their measured line properties (see COLDz utilities) and well characterized statistical correction factors (see COLDz artificial sources repo) to constrain the Schechter distribution of the CO luminosity function, i.e. the gas mass function of galaxies.

In cand_class.py we define the class of line candidates, initializing an instance of this class extracts all the appropriate, derived properties and stores them as attributes.

In ABC_COSMOS.py we use the ABC method (Approximate Bayesian Computation) to sample the probability distribution for the Luminosity function distribution Schechter parameters, accounting for the different sources of systematic uncertainty introduced by the measurement.

In plot_distribution.py we plot the derived probabilistic constraints on the Luminosity function.
