# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:40:42 2021

@author: paddy
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import datetime, timedelta
import os
import statistics as st
from matplotlib.ticker import MultipleLocator
import scipy.interpolate as spinterp
from tqdm import tqdm

def init(bstfile='C:/Users/paddy/OneDrive/DCU/ilofar/workshop/BST_tutorial/BST_data/modea/20170902_103626_bst_00X.dat', subbands=np.arange(7,495), beamlets=np.arange(0,488)):
    """
    Turn a .dat file into a Numpy arrays for data, time and frequencys

    Parameters
    ----------
    bstfile : .dat file
        This is the file that will be turned into arrays
    subbands : List, optional
        The default is np.arange(7,495).
    beamlets : List, optional
        The default is np.arange(0,488).

    Returns
    -------
    obs_start :

    """
    
    data = np.fromfile(bstfile)
    print("Number of data points:",data.shape[0])
    
    file_size = os.path.getsize(bstfile)
    print("File size:",file_size,"bytes")
    
    bit_mode = round(file_size/data.shape[0])
    print("Bitmode:",bit_mode)
    
    #length of the observation
    
    t_len = (data.shape[0]/len(subbands))
    print("Time samples:",round(t_len),"seconds")
    
    #reshape data
    
    print("Data preshape:",data.shape)
    data = data.reshape(-1,len(beamlets))
    print("Data reshape:",data.shape)
    
    #datetime objects
    global obs_start
    obs_start = bstfile[len(bstfile)-27:len(bstfile)-12]
    
    obs_start = datetime.strptime(obs_start,"%Y%m%d_%H%M%S")
    global date_format
    date_format = dates.DateFormatter("%H:%M")
    print("Start time:",obs_start)
    obs_len = timedelta(seconds = t_len)
    obs_end = obs_start + obs_len
    t_lims = [obs_start, obs_end]
    t_lims = dates.date2num(t_lims)
    print("End time:",obs_end)
    
    #time array
    
    global t_arr 
    t_arr = np.arange(0,data.shape[0])
    t_arr = t_arr*timedelta(seconds=1)
    t_arr = obs_start+t_arr
    t_arr = dates.date2num(t_arr)
    
    global data_F 
    data_F = data/np.mean(data[:100],axis=0)
    
    global sbs
    sbs = subbands
    global freqs
    freqs = sb_to_freq(sbs)
    

def sb_to_freq(sb, beamlets=np.arange(0,488)):
    
    def sb_to_freq_math(x): return ((n-1)+(x/512))*(clock/2)
    
    clock = 200 #MHz
    
    sb_3 = np.arange(54,454,2)
    sb_5 = np.arange(54,454,2)
    sb_7 = np.arange(54,290,2)

    n = 1
    freq_3 = sb_to_freq_math(sb_3)

    n = 2
    freq_5 = sb_to_freq_math(sb_5)

    n = 3
    freq_7 = sb_to_freq_math(sb_7)
    
    freq = np.concatenate((freq_3,freq_5,freq_7),axis=0)
    
    freq = freq[beamlets[0]:beamlets[-1]+1]
            
    return freq


def differential(dy,dx=1): return np.diff(dy)/dx


def cleaningprocess(data_F,c1=20,c2=2,c3=20,c4=2,int_lim=0.5):
    """
    

    Parameters
    ----------
    data_F : Numpy array
        
    c1 : The absolute limit of derivative, optional
        The default is 20.
    c2 : The limit as a multiple of standard dev of derivative, optional
        The default is 2.
    c3 : The absolute limit of 2nd derivative, optional
        DESCRIPTION. The default 20.
    c4 : The limit as a multiple of standard dev of 2nd derivative, optional
        The default is 2.
    int_lim : The upper limit of the intensity, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    datacleaned : Numpy Array
        Ready to be used as as mask for interpolateprocess().

    """
    datacleaned = data_F.copy()
    
    for s in range(len(data_F)):
        diffdata = differential(data_F[s])
        diff2data = differential(diffdata)
        stdev = st.stdev(diffdata)
        stdev2 = st.stdev(diff2data)
        
        for i in range(len(diffdata)):
            if abs(diffdata[i]) >= c1 or abs(diffdata[i]) >= stdev*c2 or data_F[s][i] <= int_lim:
                datacleaned[s][i] = np.NaN
        for i in range(len(diff2data)):
            if abs(diff2data[i]) >= c3 or abs(diff2data[i]) >= stdev2*c4:
                datacleaned[s][i] = np.NaN
                
    return datacleaned


def interpolateprocess(datacleaned):
    """
    

    Parameters
    ----------
    datacleaned : Numpy array
        Array consiting of np.NaNs which will define areas for interpolation.

    Returns
    -------
    datainterp : Numpy array
        Interpolated data array.

    """
    #filter mask
    datamasked = np.ma.masked_invalid(datacleaned)

    #interpolate data
    x, y = np.meshgrid(np.arange(0, datacleaned.shape[1]), np.arange(0, datacleaned.shape[0]))

    x1 = x[~datamasked.mask]
    y1 = y[~datamasked.mask]
    newarr = datamasked[~datamasked.mask]

    #method linear is longer than nearest
    datainterp = spinterp.griddata((x1, y1), newarr.ravel(), (x, y), method='nearest')
    return datainterp


def clean_and_interp(data_F,c1=20,c2=2,c3=20,c4=2,int_lim=0.5):
     """
    

    Parameters
    ----------
    data_F : Numpy array
        
    c1 : The absolute limit of derivative, optional
        The default is 20.
    c2 : The limit as a multiple of standard dev of derivative, optional
        The default is 2.
    c3 : The absolute limit of 2nd derivative, optional
        DESCRIPTION. The default 20.
    c4 : The limit as a multiple of standard dev of 2nd derivative, optional
        The default is 2.
    int_lim : The upper limit of the intensity, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    datainterp : Numpy Array
        Cleaned and interpolated data.

    """
    
    return interpolateprocess(cleaningprocess(data_F,c1,c2,c3,c4,int_lim))