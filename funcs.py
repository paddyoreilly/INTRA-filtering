# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:40:42 2021

@author: paddy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import datetime, timedelta
import os
import statistics as st
from matplotlib.ticker import MultipleLocator
import scipy.interpolate as spinterp
from tqdm import tqdm

def init(bstfile='C:/Users/paddy/OneDrive/DCU/ilofar/workshop/BST_tutorial/BST_data/modea/20170902_103626_bst_00X.dat', subbands=np.arange(7,495), beamlets=np.arange(0,488)):
    
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

def diffcomp(pick, sel):
    
    sel_data = pick[sel][:200]
    
    fig, xdx = plt.subplots(nrows=3,ncols=1,figsize=(10, 10), dpi=400)
    xdx2 = xdx[0].twiny()
    
    xdx[0].plot(np.arange(len(sel_data)),sel_data)
    
    if np.isnan(np.sum(sel_data)) == True:
        xdx2.scatter(freqs[:200],sel_data, marker=".")
    else:
        xdx2.plot(freqs[:200],sel_data)
    
    xdx[0].set_xlabel("Band")
    xdx2.set_xlabel("Frequency (MHz)")
    
    xdx[0].set_ylabel("Intensity")

    xdx[0].set_title("Time: " + dates.num2date(t_arr[sel]).strftime("%a %H:%M:%S") + " (Time sample: " + str(sel) + ")")
    
    diffdata = differential(sel_data)
    
    if np.isnan(np.sum(sel_data)) == False:
        dev = st.stdev(diffdata)
        xdx[1].text(100,round(max(diffdata),5)/2,"σ = "+str(round(dev,2)),fontsize=10)
    
    if np.isnan(np.sum(sel_data)) == True:
        xdx[1].scatter(np.arange(len(diffdata)),diffdata, marker=".")
    xdx[1].plot(np.arange(len(diffdata)),diffdata)

    xdx[1].set_xlabel("Band")
    xdx[1].set_ylabel("dt/dB")

    xdx[2].imshow(pick.T[:200], aspect="auto", vmin=np.percentile(pick,2), vmax=np.percentile(pick,98), origin='lower')

    xdx[2].axvline(sel, color="r")

    xdx[2].set_xlabel("Time sample")
    xdx[2].set_ylabel("Band")

    plt.show()
    
def diff2comp(pick, sel):
    
    sel_data = pick[sel][:200]
    
    fig, xdx = plt.subplots(nrows=4,ncols=1,figsize=(10, 10), dpi=400)
    xdx2 = xdx[0].twiny()
    
    xdx[0].plot(np.arange(len(sel_data)),sel_data)
    
    if np.isnan(np.sum(sel_data)) == True:
        xdx2.scatter(freqs[:200],sel_data, marker=".")
    else:
        xdx2.plot(freqs[:200],sel_data)
    
    xdx[0].set_xlabel("Band")
    xdx2.set_xlabel("Frequency (MHz)")
    
    xdx[0].set_ylabel("Intensity")

    xdx[0].set_title("Time: " + dates.num2date(t_arr[sel]).strftime("%a %H:%M:%S") + " (Time sample: " + str(sel) + ")")
    
    diffdata = differential(sel_data)
    diff2data = differential(diffdata)
    
    if np.isnan(np.sum(sel_data)) == False:
        dev = st.stdev(diffdata)
        dev2 = st.stdev(diff2data)
        xdx[1].text(100,round(max(diffdata),5)/2,"σ = "+str(round(dev,2)),fontsize=10)
        xdx[2].text(100,round(max(diff2data),5)/2,"σ = "+str(round(dev2,2)),fontsize=10)
    
    if np.isnan(np.sum(sel_data)) == True:
        xdx[1].scatter(np.arange(len(diffdata)),diffdata, marker=".")
        xdx[2].scatter(np.arange(len(diff2data)),diffdata, marker=".")
    xdx[1].plot(np.arange(len(diffdata)),diffdata)
    xdx[2].plot(np.arange(len(diff2data)),diff2data)

    xdx[1].set_xlabel("Band")
    xdx[1].set_ylabel("dt/dB")
    xdx[2].set_xlabel("Band")
    xdx[2].set_ylabel("d2t/dB2")

    xdx[3].imshow(pick.T[:200], aspect="auto", vmin=np.percentile(pick,2), vmax=np.percentile(pick,98), origin='lower')

    xdx[3].axvline(sel, color="r")

    xdx[3].set_xlabel("Time sample")
    xdx[3].set_ylabel("Band")

    plt.show()

def cleaningprocess(pick,c1=np.NaN,c2=np.NaN,c3=np.NaN,c4=np.NaN,int_lim=np.NaN):
    #c1 = absolute limit of derivative
    #c2 = limit as a multiple of standard dev of derivative
    #c3 = absolute limit of 2nd derivative
    #c4 = limit as a multiple of standard dev of 2nd derivative
    #int_lim = absolute limit of intensity
    datacleaned = pick.copy()
    
    for s in range(len(pick)):
        diffdata = differential(pick[s])
        diff2data = differential(diffdata)
        stdev = st.stdev(diffdata)
        stdev2 = st.stdev(diff2data)
        
        for i in range(len(diffdata)):
            if abs(diffdata[i]) >= c1 or abs(diffdata[i]) >= stdev*c2 or pick[s][i] <= int_lim:
                datacleaned[s][i] = np.NaN
        for i in range(len(diff2data)):
            if abs(diff2data[i]) >= c3 or abs(diff2data[i]) >= stdev2*c4:
                datacleaned[s][i] = np.NaN
                
    return datacleaned

def interpolateprocess(datacleaned):
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

def pcolormeshplot(data, sbs, a=False, title=""):
    freqs = sb_to_freq(sbs)
    g1 = 200
    g2 = 400
    
    fig, ax = plt.subplots(figsize=(15, 10), dpi=400)
    
    plt.pcolormesh(t_arr,freqs[:g1],data.T[:g1], shading='auto',
                   vmin=np.percentile(data.T,2), vmax=np.percentile(data.T,98));
    if a == True:
        plt.pcolormesh(t_arr,freqs[g1:g2],data.T[g1:g2], shading='auto',
                       vmin=np.percentile(data.T,2), vmax=np.percentile(data.T,98));
        plt.pcolormesh(t_arr,freqs[g2:],data.T[g2:], shading='auto',
                       vmin=np.percentile(data.T,2), vmax=np.percentile(data.T,98));
        
    plt.gca().invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(date_format)
    
    plt.xlabel("Time")
    plt.ylabel("Frequency (MHz)")
    
    plt.colorbar()
    plt.title(obs_start.strftime("%a %d %B %y (%Y%m%d_%H%M%S) ")+title)
    plt.show()

def pcolormeshplot_m(data, sbs, t_arr, v, a=False, title="", colorbar=False,cmap=None):
    freqs = sb_to_freq(sbs)
    g1 = 200
    g2 = 400
    
    #fig, ax = plt.subplots(figsize=(15, 10), dpi=400)
    
    plt.pcolormesh(t_arr,freqs[:g1],data.T[:g1], shading='auto',
                   vmin=np.percentile(v,2), vmax=np.percentile(v,98),cmap=cmap);
    if a == True:
        plt.pcolormesh(t_arr,freqs[g1:g2],data.T[g1:g2], shading='auto',
                       vmin=np.percentile(data.T,2), vmax=np.percentile(v,98),cmap=cmap);
        plt.pcolormesh(t_arr,freqs[g2:],data.T[g2:], shading='auto',
                       vmin=np.percentile(v,2), vmax=np.percentile(v,98),cmap=cmap);
        
    #plt.gca().invert_yaxis()
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(date_format)
    
    plt.xlabel("Time")
    plt.ylabel("Frequency (MHz)")
    
    if colorbar == True:
        plt.colorbar()
        
    plt.title(title)
    #plt.show()