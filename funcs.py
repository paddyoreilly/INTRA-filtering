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

def start(bstfile, subbands=np.arange(488)):
    """
    Turn a .dat file into a Numpy arrays for data, time and frequencys.

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
    obs_start : datetime
        The time when the observation begins.
    t_arr : Numpy array
        A 1d array of times converted from datetime.
    data_F : Numpy array
        The data array, ready to be plotted or cleaned.
    sbs : Numpy array
        A 1d array of subbands used
    freqs : Numpy array
        A 1d array of frequencys

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
    data = data.reshape(-1,len(subbands))
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
    

def sb_to_freq(sb=np.arange(488)):
    
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
    
    freq = freq[sb[0]:sb[-1]+1]
            
    return freq


def differential(dy,dx=1): return np.diff(dy)/dx


def cleaningprocess(data_F,c1=20,c2=2,c3=20,c4=2,int_lim=0):
    """
    Removes the datapoints which are outside of the limits set by the constants.
    Returns a Numpy array consisting of NaN values where the values were outside the limit.

    Parameters
    ----------
    data_F : Numpy array
        
    c1 : The absolute limit of derivative, optional
        The default is 20.
    c2 : The limit as a multiple of standard dev of derivative, optional
        The default is 2.
    c3 : The absolute limit of 2nd derivative, optional
        The default 20.
    c4 : The limit as a multiple of standard dev of 2nd derivative, optional
        The default is 2.
    int_lim : The upper limit of the intensity, optional
        The default is 0.

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
    Takes a dataset with removed datapoints, and interpolates it into a full image.
    Returns a full Numpy array.

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


def clean_I(data_F,c1=20,c2=2,c3=20,c4=2,int_lim=0):
    """
    Removes the datapoints which are outside of the limits set by the constants and interpolates it into a full image.
    Returns a full Numpy array.

    Parameters
    ----------
    data_F : Numpy array
        
    c1 : The absolute limit of derivative, optional
        The default is 20.
    c2 : The limit as a multiple of standard dev of derivative, optional
        The default is 2.
    c3 : The absolute limit of 2nd derivative, optional
         The default 20.
    c4 : The limit as a multiple of standard dev of 2nd derivative, optional
        The default is 2.
    int_lim : The upper limit of the intensity, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    datainterp : Numpy Array
        Cleaned and interpolated data.

    """
    
    return interpolateprocess(cleaningprocess(data_F,c1,c2,c3,c4,int_lim))

def pcolormeshplot(data, y, x, vdata=None, tfs=True, title="", colorbar=False,cmap=None,alpha=None,figsize=None):
    """
    Uses pyplot's pcolormesh function to plot data

    Parameters
    ----------
    data : Numpy array
        The Numpy array of the data that will be plotted.
    y : 1d array or list
        The y values that will be plotted (freqs).
    x : 1d array or list
        The x values that will be plotted (t_arr).
    vdata : Numpy array
        Used for vmax and vmin.
    tfs : Bool, optional
        If False, just plots 10-90 MHz. The default is False.
    title : String, optional
        Title of the plot. The default is "".
    colorbar : Bool, optional
        Adds a colorbar. The default is False.
    cmap : String, optional
        Change the colormap of the plot. The default is None.
    alpha : Float, optional
        Change the opacity of the plot. The default is None.
    figsize : Tuple, optional
        Sets the size of the plot in inches (400dpi). The default is None.


    Returns
    -------
    Plot.

    """
    if vdata is None:
        vdata = data
    g1 = 200
    g2 = 400
        
    if figsize is None:
        pass
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=400)
    
    plt.pcolormesh(x,y[:g1],data.T[:g1], shading='auto',
                   vmin=np.percentile(vdata,2), vmax=np.percentile(vdata,98),cmap=cmap,alpha=alpha);
    if tfs == True:
        plt.pcolormesh(x,y[g1:g2],data.T[g1:g2], shading='auto',
                       vmin=np.percentile(vdata,2), vmax=np.percentile(vdata,98),cmap=cmap,alpha=alpha);
        plt.pcolormesh(x,y[g2:],data.T[g2:], shading='auto',
                       vmin=np.percentile(vdata,2), vmax=np.percentile(vdata,98),cmap=cmap,alpha=alpha);
        
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(date_format)
    
    plt.xlabel("Time")
    plt.ylabel("Frequency (MHz)")
    
    if colorbar == True:
        plt.colorbar()
        
    plt.title(title)
    
    
def badchannelfinder(data,cut=0.75,upper_lim=20,lower_lim=0.5, histreturn=False, plot=False):
    """
    Seeks out bad channels.
    First defines a 'normal' limit. Any channel outside this limit for longer than the cutoff time is placed inside a list.

    Parameters
    ----------
    data : Numpy array
        The data array.
    cut : The cutoff point as a multiple of the time for bad bands, optional
        ie. If a band is outside the limit for 0.75 of the time it is considered bad. The default is 0.75.
    upper_lim : Upper limit for bad bands, optional
        The default is 20.
    lower_lim : Lower limit for bad bands, optional
        The default is 0.5.
    plot : Plot the data array alongside a histogram with counts outside the limit, optional
        The default is False.
    histreturn : Return the histogram of counts outside the limit as hist, optional
        The default is False.

    Returns
    -------
    bad : List
        List of bands outside the limit for longer than the cutoff.

    """
    
    if histreturn == True:
        global hist
    hist = []
    for i in range(len(data.T)):
        count = 0
        for s in range(len(data)):
            if data[s,i] >= upper_lim or data[s,i] <= lower_lim:
                count = count+1
        hist.append(count)
    
    bad = []
    for i in range(len(hist)):
        if hist[i] >= cut*len(data):
            bad.append(i)
    
    if plot == True:
        fig, ax = plt.subplots(figsize=(15,10), dpi=400)
        
        plt.subplot2grid(shape=(1,4),loc=(0,0),colspan=3)
        pcolormeshplot(data,freqs,t_arr,tfs=True,title="Bad channels are outside limit for "+str(round(cut*100))+"% of the whole observation")
        
        for i in bad:
            plt.axhline(freqs[i],c='r',alpha=0.5)
        
        plt.subplot2grid(shape=(1,4),loc=(0,3))
        plt.barh(np.arange(len(freqs)),hist,1)
        
        plt.ylim(0,len(freqs))
        plt.xlim(0,st.stdev(hist)*5+np.average(hist))
        
        plt.gca().yaxis.set_major_locator(MultipleLocator(20))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(2))
        
        plt.ylabel("Subbands")
        plt.xlabel("Counts")
        
        plt.title("Counts of Values over "+str(upper_lim)+" or under "+str(lower_lim))
        
        plt.axvline(cut*len(data),c="r")
        
        plt.tight_layout()
        plt.show()
    
    return bad


def badchannelremover(data,cut=0.75,upper_lim=20,lower_lim=0.5,bad=None):
    """
    Seeks out bad channels and removes them.
    If a list of bad channels is given, it will remove them instead of seeking the bad channels out.

    Parameters
    ----------
    data : Numpy array
        The data array.
    cut : The cutoff point as a multiple of the time for bad bands, optional
        ie. If a band is outside the limit for 0.75 of the time it is considered bad. The default is 0.75.
    upper_lim : Upper limit for bad bands, optional
        The default is 20.
    lower_lim : Lower limit for bad bands, optional
        The default is 0.5.
    bad : If you want to define the bad bands yourself, optional
        The default is None.

    Returns
    -------
    data : Numpy array
        Array with bad bands removed.

    """
    data_n = data.copy()
    
    if bad is None:
        bad = badchannelfinder(data,cut=len(t_arr)*0.75,upper_lim=20,lower_lim=0.5)
                
    for i in bad:
        data_n.T[i] = np.NaN*len(freqs)
        
    return data_n


def completeclean(data,cut=0.75,upper_lim=20,lower_lim=0.5,c1=20,c2=2,c3=20,c4=2,int_lim=0, bad=None, interpolate=True, savepath=None):
    """
    Runs the complete cleaning process, including removal of bad subbands.

    Parameters
    ----------
    data : Numpy array
        The uncleaned data array.
    cut : The cutoff point as a multiple of the time for bad bands, optional
        ie. If a band is outside the limit for 0.75 of the time it is considered bad. The default is 0.75.
    upper_lim : Upper limit for bad bands, optional
        The default is 20.
    lower_lim : Lower limit for bad bands, optional
        The default is 0.5.
    c1 : The absolute limit of derivative, optional
        The default is 20.
    c2 : The limit as a multiple of standard dev of derivative, optional
        The default is 2.
    c3 : The absolute limit of 2nd derivative, optional
         The default 20.
    c4 : The limit as a multiple of standard dev of 2nd derivative, optional
        The default is 2.
    int_lim : The upper limit of the intensity, optional
        The default is 0.
    bad : If you want to define the bad bands yourself, optional
        The default is None.
    interpolate : If you want the end result interpolated, optional
        The default is True.
    savepath : Saves the clean data to a .npy file, optional
        The default is None.
        
    Returns
    -------
    Numpy array
        Clean array with bad bands removed.

    """
    if bad is None:
        bad = badchannelfinder(data,cut,upper_lim,lower_lim)
        
    data_n = badchannelremover(cleaningprocess(data,c1,c2,c3,c4,int_lim),bad=bad)
    
    if interpolate == True:
        data_n = interpolateprocess(data_n)

    if savepath == None:
        pass
    else:
        np.save(savepath,data_n)
        
    return data_n


def starttoclean(bstfile, savepath=None, plot=False, cut=0.75,upper_lim=20,lower_lim=0.5,c1=20,c2=2,c3=20,c4=2,int_lim=0.5):
    """
    Runs the complete cleaning process from the .dat file. 
    Specify a savepath in order to save as a .npy file 
    Set plot to True to plot the end result

    Parameters
    ----------
    bstfile : .dat file
        This is the file that will be turned into arrays
    savepath : Saves the clean data to a .npy file, optional
        The default is None.
    plot : Bool, optional
        Plot the end result or not. The default is False.
    cut : The cutoff point as a multiple of the time for bad bands, optional
        ie. If a band is outside the limit for 0.75 of the time it is considered bad. The default is 0.75.
    upper_lim : Upper limit for bad bands, optional
        The default is 20.
    lower_lim : Lower limit for bad bands, optional
        The default is 0.5.
    c1 : The absolute limit of derivative, optional
        The default is 20.
    c2 : The limit as a multiple of standard dev of derivative, optional
        The default is 2.
    c3 : The absolute limit of 2nd derivative, optional
         The default 20.
    c4 : The limit as a multiple of standard dev of 2nd derivative, optional
        The default is 2.
    int_lim : The upper limit of the intensity, optional
        The default is 0.5.

    Returns
    -------
    Numpy array
        Clean array with bad bands removed.

    """
    start(bstfile)
    data_I = completeclean(data_F,cut,upper_lim,lower_lim,c1,c2,c3,c4,int_lim, savepath=savepath)
    if plot == True:
        pcolormeshplot(data_I, freqs, t_arr, tfs=True, title=obs_start.strftime("%a %d %B %y (%Y%m%d_%H%M%S)\n"), colorbar=True,cmap='Greys_r',figsize=(15,10))
        
    return data_I
        
if __name__ == '__main__':
    i = input("Input file:\n")
    o = input("Output file:\n")
    starttoclean(i, o)
