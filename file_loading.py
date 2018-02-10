import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table


def get_As( string_with_As ):
    return float(string_with_As.split('_As')[1].split('_mva')[0])

def get_filenames_PS_PC( input_As, PS_names, PC_names ):
    PS_filename_candidates = [x for x in PS_names if np.isclose(input_As,get_As(x), atol=1e-4)]
    PC_filename_candidates = [x for x in PC_names if np.isclose(input_As,get_As(x), atol=1e-4)]
    if len(PS_filename_candidates) == 1 and len(PC_filename_candidates) == 1:
        return PS_filename_candidates[0], PC_filename_candidates[0]
    else:
        print("multiple file candidates found! possibly fiducial model 1a/1b")
        return None

def get_table( data_dir, PS_directory, PC_directory, redshift_string='05' ):
    """
    args: 
        data_dir: home directory of data, this is just where we look for the 
            cosmological_parameters.txt file.
        PS_directory : where the power spectrum npy files are
        PC_directory: where the peak count npy files are
        redshift_string: this function loads in only one redshift string at a time.
    
    """
    # I'm running this locally, change to /tigress/jialiu/zack/ if you're on tigress
    #data_dir = '/home/zequnl/Projects/neutrino_mpk/'
    #PS_directory = data_dir + 'powerspectrum_noiseless'
    #PC_directory = data_dir + 'peakcounts_noiseless'

    # this code finds all the filenames and puts them in a list,
    # BUT ONLY FOR z05, we'll load in the rest separately 
    PS_names, PC_names = [], []
    for filename in os.listdir(PS_directory):
        if filename.endswith("_z" + redshift_string + ".npy"):  
            PS_names.append(filename)
    for filename in os.listdir(PC_directory):
        if filename.endswith("_z" + redshift_string + ".npy"): 
            PC_names.append(filename)
    print('Found', len(PS_names), 'PS files and', len(PC_names), 'PC files.')

    t = Table.read(data_dir + 'cosmological_parameters.txt', format='ascii')

    PS_filename_list_temp = []
    PC_filename_list_temp = []
    for row in t:
        # special cases (fiducial 1 and 2)
        if '1a(fiducial)' in row['Model']:
            PS_name = 'Om0.29997_As2.10000_mva0.00000_mvb0.00000_mvc0.00000_h0.70000_Ode0.69995_PS_50_z' +redshift_string+ '.npy'
            PC_name = 'Om0.29997_As2.10000_mva0.00000_mvb0.00000_mvc0.00000_h0.70000_Ode0.69995_PC_S_z' +redshift_string+'.npy'
        elif '1b(fiducial)' in row['Model']:
            PS_name = 'Om0.29780_As2.10000_mva0.02175_mvb0.02338_mvc0.05486_h0.70000_Ode0.69995_PS_50_z' +redshift_string+'.npy'
            PC_name = 'Om0.29780_As2.10000_mva0.02175_mvb0.02338_mvc0.05486_h0.70000_Ode0.69995_PC_S_z' +redshift_string+'.npy'
        else:
            PS_name, PC_name = get_filenames_PS_PC(row['10^9*A_s'], PS_names, PC_names)

        PS_filename_list_temp.append(PS_name)
        PC_filename_list_temp.append(PC_name)

    # put the filenames in the table for easy access
    t['PS'] = np.array(PS_filename_list_temp)
    t['PC'] = np.array(PC_filename_list_temp)

    # show the table
    return t



# get param and PC/PS utility function for many redshifts
def get_data_arrays_across_redshifts(table, data_dir, PS_directory, PC_directory, redshifts=['05'], 
                    l_min = 200, l_max = 5000, 
                    kappa_min = -0.05, kappa_max=np.inf, sky_coverage=2e4,
                                     third_variable='sigma_8(derived)' ):
    """
    wrapper function
        args: 
        data_dir: home directory of data, this is just where we look for the 
            cosmological_parameters.txt file.
        PS_directory : where the power spectrum npy files are
        PC_directory: where the peak count npy files are
        redshift_string: this function loads in only one redshift string at a time.
    
    returns: params, obsarr_PS, obsarr_PC, ell, kappa, invcov_PS, invcov_PC
    """
    # get the table, which should list the parameters for redshift 0.5. 
    # we will use this info for the other redshifts, just replacing the 0.5.
 
    PS_means = []
    PC_means = []
    PC_redshifts = [rr for rr in redshifts if not rr=='11000' ]
    
    for row in table:
        # generate a list of files to read by replacing the z05
        PS_filenames = [ row['PS'].replace('05.npy', r+'.npy') \
                        for r in redshifts ]
        PC_filenames = [ row['PC'].replace('05.npy', r+'.npy') \
                        for r in redshifts ]
        
        # loop over redshifts and append them to a list
        PS_arr = []
        PC_arr = []
        for z_ind, redshift in enumerate(redshifts):
            PS_temp = np.load( PS_directory + r'/' + PS_filenames[z_ind] )
            PC_temp = np.load( PC_directory + r'/' + PC_filenames[z_ind] )
            
            # now filter for ell and kappa
            ell = PS_temp[0,:]
            kappa = PC_temp[0,:]
            
            # SPECIAL CASE FOR CMB:
            if redshift == '11000':
                ell_filter = np.logical_and( ell > max(l_min,360), ell < l_max )
                kappa_filter = np.logical_and( kappa > kappa_min, kappa < min(kappa_max,0.2) )
            else:
                ell_filter = np.logical_and( ell > l_min, ell < l_max )
                kappa_filter = np.logical_and( kappa > kappa_min, kappa < kappa_max )
            
            PS_arr.append( ((PS_temp.T)[ell_filter]).T )
            PC_arr.append( ((PC_temp.T)[kappa_filter]).T )
            
        # debugging output : bin contributions
        if row['Model'] == '1a(fiducial)':
            print( 'PS bins', [len(bin_contrib[0,:]) for bin_contrib in PS_arr] )
            print( 'PC bins', [len(bin_contrib[0,:]) for bin_contrib in PC_arr] )
            
        # now stack them together sideways
        PS_arr = np.hstack(PS_arr)
        PC_arr = np.hstack(PC_arr)

        # first row of PS_arr is ell
        ell = PS_arr[0,:]
        PS_realizations = PS_arr[1:,:]
        # first and second row of PC are kappa, SNR
        kappa = PC_arr[0,:]
        PC_realizations = PC_arr[2:,:]

        PS_means.append( np.mean(PS_realizations, axis=0) )
        PC_means.append( np.mean(PC_realizations, axis=0) )
        
        # compute covariances
        if row['Model'] == '1a(fiducial)':
            PS_cov = np.cov(PS_realizations.T)
            PC_cov = np.cov(PC_realizations.T)
            
            nrealizations, nbins = PS_realizations.shape
            PS_correction = (nrealizations - nbins - 2) / (nrealizations - 1)
            print( 'PS nr', nrealizations, 'nb', nbins, PS_correction )
            nrealizations, nbins = PC_realizations.shape
            PC_correction = (nrealizations - nbins - 2) / (nrealizations - 1)
            print( 'PC nr', nrealizations, 'nb', nbins, PC_correction )
            
            # this 12.25/2e4 is from the LSST area divided by box, from Jia's email
            invcov_PS = PS_correction * np.linalg.inv(PS_cov * 12.25/sky_coverage)
            invcov_PC = PC_correction * np.linalg.inv(PC_cov * 12.25/sky_coverage)
            

    obsarr_PS = np.array( PS_means ) 
    obsarr_PC = np.array( PC_means ) 
    params = np.array( [table['M_nu(eV)'], table['Omega_m'], table[third_variable]] ).T

    return params, obsarr_PS, obsarr_PC, ell, kappa, invcov_PS, invcov_PC


# compute custom cov
def compute_custom_PC_cov( redshifts, PC_directory='peakcounts_cov', 
                          kappa_min = -9.61769831e-03, kappa_max=np.inf, sky_coverage = 2e4,
             fid_string = 'Om0.29997_As2.10000_mva0.00000_mvb0.00000_mvc0.00000_h0.70000_Ode0.69995_PC_S_z05_cov.npy'
    ):
    """
    This function computes a custom covariance from a separate 10000 realizations.
    Since this is a special case, everything is pretty hardcoded. It also repeats a bunch of code with the normal routines.
    """
    
    # replaces _z05_ with _z10_ etc.
    #CMB_filestring = 'peakcounts_noiseless/Om0.29997_As2.10000_mva0.00000_mvb0.00000_mvc0.00000_h0.70000_Ode0.69995_PC_S_z11000.npy'
    
    PC_filenames = [ PC_directory + r'/' + fid_string.replace('_z05_', '_z' + r + '_') 
                    for r in redshifts ]
  
    # loop over redshifts and append them to a list
    PS_arr = []
    PC_arr = []
    for z_ind, redshift in enumerate(redshifts):
        PC_temp = np.load( PC_filenames[z_ind] )

        # now filter for ell and kappa
        kappa = PC_temp[0,:]
        kappa_filter = np.logical_and( kappa > kappa_min, kappa < kappa_max )
        PC_arr.append( ((PC_temp.T)[kappa_filter]).T )

        print( 'PC bins', [len(bin_contrib[0,:]) for bin_contrib in PC_arr] )

    # now stack them together sideways
    PC_arr = np.hstack(PC_arr)

    # first and second row of PC are kappa, SNR
    kappa = kappa[kappa_filter]
    PC_realizations = PC_arr[2:,:]
    PC_cov = np.cov(PC_realizations.T)
    nrealizations, nbins = PC_realizations.shape
    PC_correction = (nrealizations - nbins - 2) / (nrealizations - 1)
    print( 'PC nr', nrealizations, 'nb', nbins, PC_correction )        

    # this 12.25/2e4 is from the LSST area divided by box, from Jia's email
    invcov_PC = PC_correction * np.linalg.inv(PC_cov * 12.25/sky_coverage)
    return invcov_PC

    