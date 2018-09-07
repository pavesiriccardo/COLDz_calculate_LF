import numpy as np,matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl


mpl.rcParams['font.size']=24  #normal is 12
mpl.rcParams['lines.linewidth']=2  #normal is 1
mpl.rcParams['axes.linewidth'] = 2

plt.style.use('classic')

fig, ax = plt.subplots(1,1,figsize=(12,9))
fig.subplots_adjust(hspace=0,wspace=0)


def logschecter(logLprime,logLstar,logphistar,alpha):
	'''
	Calculate the log of the Schechter distribution for a particular line luminosity and distribution parameters.

	Parameters

	logLprime:float
		The log10 of the L' line luminosity

	logLstar,logphistar,alpha:(float,float,float)
		The Schechter distribution parameters 

	Return

	The log10 of the distribution density

	'''
	return logphistar+alpha*(logLprime-logLstar)-10**(logLprime-logLstar)/np.log(10)+np.log10(np.log(10))



#Combine all the samples in equal proportions. 
samp_norm=np.loadtxt('/data2/common/goodsN/SNRanalysis_57w/MF3D/ABC_samples/mysamples_latest_normal')
samp_uni=np.loadtxt('/data2/common/goodsN/SNRanalysis_57w/MF3D/ABC_samples/mysamples_latest_uniform')
samp_tot=np.zeros((10000,3))
samp_tot[:5000]=samp_norm[:5000]
samp_tot[5000:]=samp_uni[:5000]
samp=samp_tot

#For each parameter sample, calculate the distribution function values over the full x-axis.
Lprimerange=np.arange(9.5,12,.01)
all_phi=[]
for sam in samp:
	all_phi.append(logschecter(Lprimerange,sam[0],sam[1],sam[2]))

all_phi=np.array(all_phi)
xaxis=np.array(list(Lprimerange)*all_phi.shape[0])

#Make a hex-plot of the density of distribution curves to show the probability density.
hexres=ax.hexbin(xaxis,all_phi.flatten(),cmap='Greens',norm=colors.PowerNorm(.5),gridsize=(int(.15*Lprimerange.shape[0])-1,int(1.5*Lprimerange.shape[0])-1),mincnt=2)


#Plot the median parameter value.
ax.plot(Lprimerange,logschecter(Lprimerange,np.percentile(samp[:,0],50),np.percentile(samp[:,1],50),np.percentile(samp[:,2],50)),color='k')
#Plot the vertical 95th percentile parameter value.
ax.plot(Lprimerange, np.percentile(all_phi,95,axis=0),color='k')
#Plot the vertical 5th percentile parameter value.
ax.plot(Lprimerange, np.percentile(all_phi,5,axis=0),color='k')

#Make the legend.
percent_line = mlines.Line2D([], [], color='k',markersize=15,label='5th, 50th, 95th\npercentiles',lw=2)
saint_line = mlines.Line2D([], [], color='r',markersize=15, label='Saintonge et al. 2017\n(z=0 LF)')
ax.legend(handles=[saint_line,percent_line],loc='upper right',frameon=False)

ax.xaxis.set_minor_locator(MultipleLocator(.1))
ax.yaxis.set_minor_locator(MultipleLocator(.1))
ax.tick_params(which='major',length=8,width=1.5)
ax.tick_params(which='minor',length=4,width=1.5)

ax.set_xlim(8.6,12.1)
ax.set_ylim(-5.9,-1.75)
ax.set_xlabel('log $L\'_{ \mathrm{CO}(1-0)}$ [$\mathrm{ K \, km \, s^{-1} \, pc^2}$]',fontsize=28)
ax.set_ylabel('log $\Phi_{\mathrm{ CO}}$ [$\mathrm{Mpc^{-3} \, dex^{-1}}$]',fontsize=28)

#Plot the local gaalxies measurement by Saintonge.
saint_sch=np.load('Saintonge_data.npy')
ax.plot(saint_sch[:,0],saint_sch[:,1],color='r',zorder=2)


plt.savefig('ABC_Schechter_distribution.pdf',bbox_inches='tight')


