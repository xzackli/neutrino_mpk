import os
import numpy as np 
import matplotlib.pyplot as plt

# inline magic changes figsize, set it again here
import matplotlib
matplotlib.rcParams['figure.figsize'] = 8,6 
default_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color'] # get default colors

from astropy.table import Table
from astropy.io import ascii
import math
import pathos.multiprocessing as mp

# get all the names of the cosmology folders
folders = os.listdir('data/May_stats/')

# helper function for extracting the numerical value of As from a filename
def get_As( string_with_As ):
    return float(string_with_As.split('_As')[1].split('_mva')[0])

# read in list of simulation cosmo parameters
table = Table.read('cosmological_parameters.txt', format='ascii')

# now combine tables
new_filename_row = []
for row in table:
    # match based on 10^9*A_s column name
    filename_candidates = [x for x in folders 
                           if np.isclose(row['10^9*A_s'],get_As(x), atol=1e-4)]
    
    if '1a(fiducial)' in row['Model']:
        fname = 'Om0.29997_As2.10000_mva0.00000_mvb0.00000_mvc0.00000_h0.70000_Ode0.69995'
    elif '1b(fiducial)' in row['Model']:
        fname = 'Om0.29780_As2.10000_mva0.02175_mvb0.02338_mvc0.05486_h0.70000_Ode0.69995'
    else:
        # make sure we have exactly one match
        assert len(filename_candidates) == 1
        fname = filename_candidates[0]
        
    new_filename_row.append( fname )
    
table['filename'] = new_filename_row
table.write('parameters.table', format='ascii', overwrite=True)

# dictionary for getting the appropriate ng for a redshift
ng_dict = {
    '05' : '08.83',
    '10' : '13.25',
    '15' : '11.15',
    '20' : '07.36',
    '25' : '04.26',
    '10_ng40' : '40.00'
}

def get_meanstack(observable_name, noisy, redshift, smoothing,  
                  bin_min, bin_max, binning='050', binscale='', bin_center_row = 0, 
                  start_index=0, end_index=5000):
    """
    observable_name (str) : 'PS' or 'Peaks'
    noisy (str) : 'K' or 'KN' for noiseless and noisy respectively
    redshift (string): ex. '05'
    smoothing (string) : s0.00
    binning (string) : number of bins
    binscale (string) : lin or log
    bin_center_row (int) : refers to if we want to use the kappa (use integer 0) 
                           or SN (use the integer 1)
    returns: tuple,
        (array (either ell or kappa), array(mean_stack))
        where mean_stack is 101 rows, stacked PS or PC
    """
    
    mean_stack_list = []
    
    for row in table:
        
        # file wrangling
        if row['Model'] == '1a(fiducial)':
            modifier = '/box1' # use box1 for means, use box5 for covariance
        else: 
            modifier = ''
            
        if redshift == '10_ng40':
            fname = '%s_%s_s%s_z%.2f_ng%s_b%s%s' % (observable_name, noisy, smoothing, 
                                                      1.0, ng_dict[redshift], 
                                                    binning, binscale)
            meandir = 'data/May_stats/%s/Maps%s/%s_Mean.npy' % (row['filename'] + modifier, 
                                            redshift, fname)
            fdir = 'data/May_stats/%s/Maps%s/%s.npy' % (row['filename'] + modifier, 
                                            redshift, fname)
        else:
            fname = '%s_%s_s%s_z%.2f_ng%s_b%s%s' % (observable_name, noisy, smoothing, 
                                                      float(redshift)/10, ng_dict[redshift], 
                                                    binning, binscale)
            meandir = 'data/May_stats/%s/Maps%s/%s_Mean.npy' % (row['filename'] + modifier, 
                                            redshift, fname)
            fdir = 'data/May_stats/%s/Maps%s/%s.npy' % (row['filename'] + modifier, 
                                            redshift, fname)
        
        # load in the file and split it up into meaningful stuff
#         obs_array_temp = np.load(meandir)
#         #if PS, ells are in first row
#         if observable_name == 'PS':
#             bin_centers = obs_array_temp[0]
#             mean_arr = obs_array_temp[1]
#         elif observable_name == 'Peaks':
#             bin_centers = obs_array_temp[bin_center_row]
#             mean_arr = obs_array_temp[2]

        obs_array_temp = np.load(fdir)
        if observable_name == 'PS':
            bin_centers = obs_array_temp[0]
            mean_arr = np.mean(obs_array_temp[1+start_index:end_index],axis=0)
        elif observable_name == 'Peaks':
            bin_centers = obs_array_temp[bin_center_row]
            mean_arr = np.mean(obs_array_temp[2+start_index:end_index],axis=0)
        
        
        filter_for_bins = np.logical_and(bin_min<bin_centers, bin_centers<bin_max)
        bin_centers = bin_centers[filter_for_bins]
        mean_arr = (mean_arr.T[filter_for_bins]).T 
        # add the model to the stack
        mean_stack_list.append(mean_arr)
    
    # return the stack of means, ordered like the table is
    return bin_centers, np.vstack(mean_stack_list)

def get_meanstack_multiz(observable_name, noisy, redshifts, smoothing,  
                         bin_min, bin_max, binning='050', binscale='', bin_center_row=0, start_index=0, end_index=5000):
    """
    get a horizontally stacked meanstack for multiple redshifts
    """
    
    bin_center_list = []
    stack_list = []
    for redshift in redshifts:
        # pass all arguments
        bin_centers, meanstack = get_meanstack(observable_name=observable_name, noisy=noisy, 
                                                redshift=redshift, smoothing=smoothing,  
                                                bin_min=bin_min, bin_max=bin_max, binning=binning, 
                                                binscale=binscale, bin_center_row=bin_center_row,
                                                start_index=start_index, end_index=end_index)
        bin_center_list.append( bin_centers )
        stack_list.append( meanstack )
        
    return np.hstack(bin_center_list), np.hstack(stack_list)

def get_invcov(observable_name, noisy, redshifts, smoothing, bin_min, bin_max,
               binning='050', binscale='', bin_center_row = 0, sky_coverage=1e4, verbose=True):
    """
    We only use the fiducial model's covariance.
    
    redshifts = list of redshifts to use
    """
    
    bin_center_list = []
    realization_list = []
    
    for redshift in redshifts:
        modifier = '/box5' # use box1 for means, use box5 for covariance
        if redshift == '10_ng40':
            fname = '%s_%s_s%s_z%.2f_ng%s_b%s%s' % (observable_name, noisy, smoothing, 
                                                      1.0, ng_dict[redshift], binning, binscale)
            fdir = 'data/May_stats/%s/Maps%s/%s.npy' % (table[0]['filename'] + modifier, 
                                                    redshift, fname)
        else:
            fname = '%s_%s_s%s_z%.2f_ng%s_b%s%s' % (observable_name, noisy, smoothing, 
                                                      float(redshift)/10, ng_dict[redshift], binning, binscale)
            fdir = 'data/May_stats/%s/Maps%s/%s.npy' % (table[0]['filename'] + modifier, 
                                                    redshift, fname)
            
        obs_array_temp = np.load(fdir)

        if observable_name == 'PS':
            bin_centers = obs_array_temp[0]
            realizations = obs_array_temp[1:]
        elif observable_name == 'Peaks':
            bin_centers = obs_array_temp[bin_center_row]
            realizations = obs_array_temp[2:]
            
        # filter out the bins we don't want, and then add to lists
        filter_for_bins = np.logical_and(bin_min<bin_centers, bin_centers<bin_max)
        bin_centers = bin_centers[filter_for_bins]
        realizations = (realizations.T[filter_for_bins]).T 
        bin_center_list.append(bin_centers)
        realization_list.append(realizations)
    
    realizations_stacked = np.hstack(realization_list)
    # now compute covariance
    cov = np.cov(realizations_stacked.T)

    nrealizations, nbins = realizations_stacked.shape
    bin_correction = (nrealizations - nbins - 2) / (nrealizations - 1)
    sky_correction = 12.25/sky_coverage
    
    if verbose: print('nr', nrealizations, 'nb', nbins, 
                      'bin', bin_correction, 'sky',sky_correction )

    # this 12.25/2e4 is from the LSST area divided by box, from Jia's email
    invcov = bin_correction * np.linalg.inv(cov * sky_correction)
    
    
    return invcov

from scipy import interpolate

class Interp:

    def __init__(self,obsarr, cosmo_params, invcov, fiducial_model,
                 function='multiquadric', smooth=0.0, weights=[0.2, 0.045, 0.1]):
        
        weighted_params = np.array(cosmo_params).copy()
        norm_mat = np.repeat( np.array([weights]).T, axis=1, repeats=cosmo_params.shape[-1] )
        weighted_params /= norm_mat
        
        self.invcov = invcov
        self.fid = fiducial_model
        self.weights = np.array(weights)
        
#         print(obsarr.shape)
        # create a list of Rbf for each independent mode
        spline_interps = [ interpolate.Rbf(*weighted_params, model, 
                                           function=function, smooth=smooth)
                          for model in obsarr.T ]
        
        # create a function that applies interpolator to the parameters given, for each mode
        self.interp_func = lambda params: np.array([ii(*(params/self.weights)) for ii in spline_interps])
        

    def P(self, parameter_input):
        dm = self.fid - self.interp_func( parameter_input ) # d - mu
        return np.exp( -0.5 * np.dot(dm.T,np.dot(self.invcov,dm)) )
    
        

        
        
        