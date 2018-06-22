import numpy as np
from scipy import *
from scipy import interpolate, stats

######## steps
#### each file has dimension 1001 x Nbins, where the first row is the bin center, the rest 1000 are the different realizations, and Nbins is the number of bins for the Powerspec or Peak counts. 
#### for PC files it has 1002 rows, where 1st row is the kappa bins, 2nd row is SNR bin center.
#### (1) compute the average Powerspec(PS)/Peak counts(PC) for each of the 100 model
#### (2) build covariance using the 1000 realizations of the massless model
#### (3) build interpolator using the buildInterpolator function using the massless fiducial model
#### (4) write down a probability function, where P=exp(-0.5*(d-mu).T*C_inverse*(d-u))
#### where d is the average PS/PC from the fiducial model (M_nu = 0.1 eV)
#### (5) compute a cube of probability
#### (6) marginalize over 1 dimension and draw a contour on the probability plane, we want to compare the plane from PS vs PC, and see how much we can constraint M_nu

###### files
#### 'powerspecrum' folder: power spectra for all 101  models, in .npy format, open by np.load(file)
#### 'peakcounts' folder: same but for peak counts
####  cosmological_parameters.txt - parameters for the 101 models
####  note the PS/PC file names are slightly different than the parameter files, since they used different definition for Omega_m (in PS/PC, Omega_m=Omega_cdm+Omega_c, while parameter files Omega_m=Omega_cdm+Omega_c+Omega_nu), but the best way to match them is using A_s, which should be the same in both cases (but note 2 fiducial models both have A_s=2.1, but one has M_nu = 0)

######### begin: build interpolator ###############
def buildInterpolator(obs_arr, cosmo_params):
    '''Build an interpolator:
    Input:
    (1) obs_arr has dimension (Npoints, Nbin), where Npoints = # of cosmological models (=100 here), and Nbins is the number of bins
    (2) cosmo_params has dimension (Npoints, Nparams), currently Nparams is hard-coded to be 3 (M_nu, Omega_m, A_s)
    
    Output:
    spline_interps
    
    Usage:
    spline_interps = buildInterpolator(obs_arr, cosmo_params)
    spline_interps(_nu, Omega_m, A_s)
    '''
    param1, param2, param3 = cosmo_params.T
    spline_interps = list()
    for ibin in range(obs_arr.shape[-1]):
        model = obs_arr[:,ibin]
        ### Zack: I am using interpolate.Rbf here, but you can also try other ones, check https://docs.scipy.org/doc/scipy/reference/interpolate.html
        iinterp = interpolate.Rbf(param1, param2, param3, model)
        spline_interps.append(iinterp)
    def interp_cosmo (params):
        '''Interpolate the powspec for certain param.
        Params: list of 3 parameters = (M_nu, Omega_m, A_s)
        '''
        mm, wm, sm = params
        gen_ps = lambda ibin: spline_interps[ibin](mm, wm, sm)
        ps_interp = array(map(gen_ps, range(obs_arr.shape[-1])))
        ps_interp = ps_interp.reshape(-1,1).squeeze()
        return ps_interp
    return interp_cosmo


def findlevel (H):
    '''Find 68%, 95%, 99% confidence level for a probability 2D plane H. I sort the pixels, and count from highest probability pixel, until I accumulated 68%, 95%, 99% of the probability, and record the values. With these values, you can draw contours on H for 1,2,3 sigmas.
    
    Input:
    H is a probability plane (marginalized over the 3rd parameter)
    
    Output:
    V = [v68, v95, v99]
    This is the 3 values in H that gives the 68%, 95%, 99% confidence level
    '''
    H[isnan(H)]=0 ## remove nan's
    H /= float(sum(H)) ## normalize the plane to 1
    
    idx = np.argsort(H.flat)[::-1] ## sort H from high to low, this is the index
    H_sorted = H.flat[idx] ## sorted H
    H_cumsum = np.cumsum(H_sorted) ## cumulated sum at each pixel
    ### find the nearest pixel that matches cumulated sum of 1,2,3 sigmas
    idx68 = where(abs(H_cumsum-0.683)==amin(abs(H_cumsum-0.683)))[0]    
    idx95 = where(abs(H_cumsum-0.955)==amin(abs(H_cumsum-0.955)))[0]
    idx99 = where(abs(H_cumsum-0.997)==amin(abs(H_cumsum-0.997)))[0]
    v68 = float(H.flat[idx[idx68]])
    v95 = float(H.flat[idx[idx95]])
    v99 = float(H.flat[idx[idx99]])
    V = [v68, v95, v99]
    return V

