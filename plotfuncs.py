# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:31:22 2021

@author: paddy
"""

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