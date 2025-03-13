import numpy as np
import ROOT as rt
import matplotlib.pyplot as plt
from array import array

std_color_list = [1, 2, 4, 8, 6, 28, 43, 7, 25, 36, 30, 40, 42, 49, 46, 38, 32, 800, 600, 900, 870, 840]
def create_TH2D(sample, name='h', title=None, binning=[None, None, None, None, None, None], weights=None, axis_title = ['','', '']):
    if title is None:
        title = name
    if (len(sample) == 0):
        for i in range(len(binning)):
            if binning[i] == None:
                binning[i] = 1
    else:
        if binning[1] is None:
            binning[1] = min(sample[:,0])
        if binning[2] is None:
            binning[2] = max(sample[:,0])
        if binning[0] is None:
            bin_w = 4*(np.percentile(sample[:,0],75) - np.percentile(sample[:,0],25))/(len(sample[:,0]))**(1./3.)
            if bin_w == 0:
                bin_w = 0.5*np.std(sample[:,0])
            if bin_w == 0:
                bin_w = 1

            binning[0] = int((binning[2] - binning[1])/bin_w)

        if binning[4] is None:
            binning[4] = min(sample[:,1])
        if binning[5] == None:
            binning[5] = max(sample[:,1])
        if binning[3] == None:
            bin_w = 4*(np.percentile(sample[:,1],75) - np.percentile(sample[:,1],25))/(len(sample[:,1]))**(1./3.)
            if bin_w == 0:
                bin_w = 0.5*np.std(sample[:,1])
            if bin_w == 0:
                bin_w = 1
            binning[3] = int((binning[5] - binning[4])/bin_w)
    if len(binning)==6:
        h = rt.TH2D(name, title, binning[0], binning[1], binning[2], binning[3], binning[4], binning[5])
    else:
        h = rt.TH2D(name, title, binning[-2]-1, array('f',binning[:binning[-2]]), binning[-1]-1, array('f', binning[binning[-2]:-2]));
    x = sample[:,0]
    y = sample[:,1]
    counts, _, _ = np.histogram2d(x, y, bins=np.array([binning[0], binning[3]]), range=np.array([[binning[1], binning[2]], [binning[4], binning[5]]]), weights = weights)
    for i in range(1, h.GetXaxis().GetNbins()+1):
        for j in range(1, h.GetYaxis().GetNbins()+1):
            h.SetBinContent(i,j,counts[i-1,j-1])


    h.SetXTitle(axis_title[0])
    h.SetYTitle(axis_title[1])
    h.SetZTitle(axis_title[2])
    h.binning = binning
    return h

def create_TH1D(x, name='h', title=None, binning=[None, None, None], weights=None, h2clone=None, axis_title = ['',''], bin_list=False):
    if title is None:
        title = name
    if h2clone == None:
        if binning[1] is None:
            binning[1] = min(x)
        if binning[2] is None:
            if ((np.percentile(x, 95)-np.percentile(x, 50))<0.2*(max(x)-np.percentile(x, 95))):
                binning[2] = np.percentile(x, 90)
            else:
                binning[2] = max(x)
        if binning[0] is None:
            bin_w = 4*(np.percentile(x,75) - np.percentile(x,25))/(len(x))**(1./3.)
            if bin_w == 0:
                bin_w = 0.5*np.std(x)
            if bin_w == 0:
                bin_w = 1
            binning[0] = int((binning[2] - binning[1])/bin_w) + 5

        if len(binning) > 3 or bin_list:
            h = rt.TH1D(name, title, len(binning)-1, array('f',binning))
        else:
            h = rt.TH1D(name, title, binning[0], binning[1], binning[2])
    else:
        h = h2clone.Clone(name)
        h.SetTitle(title)
        h.Reset()
    if len(binning)>3 or bin_list:counts, _ = np.histogram(x, bins=binning, weights = weights)
    else: counts, _ = np.histogram(x, bins=binning[0], range=np.array([binning[1], binning[2]]), weights = weights)
    for i in range(1, h.GetXaxis().GetNbins()+1):
        h.SetBinContent(i,counts[i-1])
    h.SetXTitle(axis_title[0])
    h.SetYTitle(axis_title[1])
    h.binning = binning
    return h

def create_TGraph(x,y,ex=[],ey=[], axis_title = ['','']):
    x = array("d", x)
    y = array("d", y)
    ex = array("d", ex)
    ey = array("d", ey)
    if not len(x) == len(y):
        print("length of x and y are not equal!")
    if not len(ex)==len(ey):
        print("length of ex and ey are not equal!")
    if len(ex)>0 and not len(x)==len(ex):
        print("leng of ex and x are not equal!")

    if len(ex)==0:gr = rt.TGraph(len(x),x,y)
    else: gr = rt.TGraphErrors(len(x),x,y,ex,ey)
    if len(axis_title) == 2:
    	gr.GetXaxis().SetTitle(axis_title[0])
    	gr.GetYaxis().SetTitle(axis_title[1])
    return gr

