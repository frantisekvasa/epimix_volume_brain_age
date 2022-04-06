# -*- coding: utf-8 -*-
"""

@authors: 
    Harriet Hobday (harriet.hobday@kcl.ac.uk)
    Frantisek Vasa (fdv247@gmail.com)
    (2020-2022)
    
"""

# %% import required libraries and define data directories

# general
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from sklearn import linear_model
from sklearn.metrics import median_absolute_error

# general - plotting
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sb
import ptitprince as pt

# neuroimaging
import nibabel as nib
import nilearn as nl
import nilearn.plotting

# home directory
import os
from pathlib import Path
home_dir = str(Path.home()) # home directory

# paths to data directories
demog_dir = home_dir+'/Desktop/brainager_demog'
brainager_dir = home_dir+'/Data/brainager_processed' # can be downloaded from: https://doi.org/10.6084/m9.figshare.18128225

# directory to save figures
plot_dir = home_dir+'/Desktop/epimix_brainager_plots'
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

# flag to determine whether figures are saved
save_fig = True

# change plot font
plt.rcParams["font.family"] = "arial"

# plotting parameters
# general
lbs = 20 # label size
lgs = 16 # legend size
axs = 16 # axis size
# heatmaps
hm_lbs = 15 # label size
hm_lgs = 13 # legend size
hm_axs = 11 # axis size

# %% function for plotting a masked brain map
# fine-tuned from nilearn function "plotting.plot_img"

# img_vec       image vector
# mask_vec      mask vector
# img_shape     image shape
# img_affine    image affine
# cmap          colormap
# clim          colormap limits

def plot_nl_image_masked(img_vec,mask_vec,img_shape,img_affine,cmap,clim=None,*line_args,**line_kwargs):
    if clim is None:
        #clim = (min(img_vec[mask_vec==1]),max(img_vec[mask_vec==1]))
        clim = (min(img_vec[mask_vec==1]),np.percentile(img_vec[mask_vec==1],95))
    # i) edit image and colorbar to map background to black
    img_masked = np.ones(img_vec.size)*(clim[0]-1); img_masked[mask_vec==1] = img_vec[mask_vec==1]
    cmap_under = cm.get_cmap(cmap, 256); cmap_under.set_under('white')
    # ii) convert image to nii and plot
    img_masked_nii = nib.Nifti1Image(np.reshape(img_masked,img_shape),affine=img_affine)
    nl.plotting.plot_img(img_masked_nii,colorbar=True,cmap=cmap_under, vmin=clim[0], vmax=clim[1],*line_args,**line_kwargs)

# %%

"""

Demographic data

"""

# %% subject IDs and demographics

# read in ID's of participants with epimix_scans
sub_epi = np.genfromtxt(demog_dir+'/brainager_epimix_ids.txt',dtype='str')
ns_epi = len(sub_epi) # number of participants

# read in ID's of participants with t1_scans
sub_t1 = np.genfromtxt(demog_dir+'/brainager_t1_ids.txt',dtype='str')
ns_t1 = len(sub_t1) # number of participants

# demographics
demog = pd.read_excel(demog_dir+'/brainager_ADAPTA_TINKER_COGNISCAN_demog.xlsx').to_numpy()    # demographics

# reorder demog array to match sub_epi IDs
if not all(demog[:,0].astype(str)==sub_epi):
    # determine reordering ID
    ord_id = np.zeros(ns_epi,dtype=int)
    for i in range(ns_epi):
        ord_id[i] = np.where(demog[:,0].astype(str)==sub_epi[i])[0]
    # reorder demographics array
    demog = demog[ord_id,:]
    del ord_id

# extract ID, age and sex as separate variables
sub = demog[:,0].astype(str)                                    # ID
age = np.round(demog[:,1].astype(float),0).astype(int)          # age
sex = demog[:,2].astype(str)                                    # sex

# check that IDs match
#all(sub==sub_epi)

# extract numerical IDs of position of the subset of subjects with T1 scans relative to all subjects
ids_t1_subset = np.zeros(ns_t1,dtype=int)
for i in range(ns_t1):
    ids_t1_subset[i] = np.where(sub_t1[i]==sub_epi)[0]

# check that this worked
#all(sub[ids_t1_subset]==sub_t1)

# ...these variables can be used to extract lists for age, demographics etc
# e.g. the age list of subjects with t1 scans is age[ids_t1_subset]

# %% demographic summary plot - histogram

#bins = np.linspace(min(age), max(age), int(ns/5))
bin_step = 2.5
bins = np.arange(17.5, 62.5, step=bin_step)

# ### all participants      
# plt.figure()
# plt.hist([age[sex=='M'],age[sex=='F']], bins, label=['M (N = '+str(sum(sex=='M'))+')','F (N = '+str(sum(sex=='F'))+')'],color=['orange','purple'], edgecolor='black')
# plt.legend(loc='upper right',prop={'size': lgs})
# plt.ylim([0,15])
# plt.xlabel('age (years)',size=lbs); plt.ylabel('# participants',size=lbs)
# plt.xticks(size=axs); plt.yticks(size=axs);
# if save_fig: plt.savefig(plot_dir+'age_dist_epimix.svg',bbox_inches='tight')

# ### participants with t1 scans
# plt.figure()
# ax = plt.hist([age[ids_t1_subset][sex[ids_t1_subset]=='M'],age[ids_t1_subset][sex[ids_t1_subset]=='F']], bins, label=['M (N = '+str(sum(sex[ids_t1_subset]=='M'))+')','F (N = '+str(sum(sex[ids_t1_subset]=='F'))+')'],color=['orange','purple'], edgecolor='black')
# plt.legend(loc='upper right',prop={'size': lgs})
# plt.ylim([0,15])
# plt.xlabel('age (years)',size=lbs); plt.ylabel('# participants',size=lbs)
# plt.xticks(size=axs); plt.yticks(size=axs);
# if save_fig: plt.savefig(plot_dir+'/age_dist_epimix_and_t1.svg',bbox_inches='tight')

## epimix and t1 scans combined
# epimix+t1 bin counts
hist_both_m,_ = np.histogram(age[ids_t1_subset][sex[ids_t1_subset]=='M'],bins=bins)
hist_both_f,_ = np.histogram(age[ids_t1_subset][sex[ids_t1_subset]=='F'],bins=bins)
# epimix only bin counts
hist_epi_m,_ = np.histogram(age[np.setdiff1d(np.arange(0,ns_epi),ids_t1_subset)][sex[np.setdiff1d(np.arange(0,ns_epi),ids_t1_subset)]=='M'],bins=bins)
hist_epi_f,_ = np.histogram(age[np.setdiff1d(np.arange(0,ns_epi),ids_t1_subset)][sex[np.setdiff1d(np.arange(0,ns_epi),ids_t1_subset)]=='F'],bins=bins)
# bar chart parameters
fig, ax = plt.subplots()
bin_w = 0.85
shift_m = (bin_step-2*bin_w)/2
shift_f = shift_m+bin_w
# bar chart - m
ax.bar(bins[:-1]+shift_m, hist_both_m, width=bin_w, align='edge', color='royalblue', edgecolor='black') 
ax.bar(bins[:-1]+shift_m, hist_epi_m, width=bin_w, align='edge', bottom=hist_both_m, color='darkred', edgecolor='black') 
# bar chart - f
ax.bar(bins[:-1]+shift_f, hist_both_f, width=bin_w, align='edge', color='indigo', edgecolor='black') 
ax.bar(bins[:-1]+shift_f, hist_epi_f, width=bin_w, align='edge', bottom=hist_both_f, color='mediumpurple', edgecolor='black') 
# labels etc
plt.xlabel('Age (years)',size=lbs); plt.ylabel('# Participants',size=lbs)
plt.xticks(size=axs); plt.yticks(size=axs);
if save_fig: plt.savefig(plot_dir+'/age_dist.svg',bbox_inches='tight')

# %%

"""

Processing time

"""

# %% Load processing times
# ... and simultaneously convert from seconds to minutes

# T1
time_t1 = np.zeros(ns_t1)
for i in range(ns_t1):
    time_t1[i] = int(np.genfromtxt(brainager_dir+'/'+sub_t1[i]+'/t1/brainager_runtime_sec.txt',dtype='str')[1])/60

# EPImix
time_epi = np.zeros(ns_epi)
for i in range(ns_epi):
   time_epi[i] = int(np.genfromtxt(brainager_dir+'/'+sub_epi[i]+'/epimix/brainager_runtime_sec.txt',dtype='str')[1])/60

# T1-FoV
time_t1_fov = np.zeros(ns_t1)
for i in range(ns_t1):
    time_t1_fov[i]= int(np.genfromtxt(brainager_dir+'/'+sub_t1[i]+'/t1_epifov/brainager_runtime_sec.txt',dtype='str')[1][1])/60

# %% raincloud plot

# variable names and colours
var_colours = ['indigo', 'royalblue', 'mediumpurple']
var_names = ['T$_1$-w', 'EPImix T$_1$-w', 'T$_1$-w FoV$_{EPI}$']

# combine data
dx = list( np.concatenate(( np.repeat(0,len(time_t1)), np.repeat(1,len(time_epi)), np.repeat(2,len(time_t1_fov)) )) )
dy = list( np.concatenate(( time_t1, time_epi, time_t1_fov )) )

# plot
f, ax = plt.subplots(figsize=(8, 3))
ax=pt.RainCloud(x = dx, y = dy, palette = var_colours, bw = .4, width_viol = .6, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(var_names, size=lbs); plt.xticks(fontsize=axs)
plt.xlabel('Processing time (minutes)', size=lbs); f.tight_layout()
if save_fig: plt.savefig(plot_dir+'/processing_time.svg',bbox_inches='tight')

# %%

"""

Tissue volumes - Global

"""

# %% read in volume files 
# csv files contain tissue volumes as [file_name, GM, WM, CSF]

# ids of different tissues in volume variables
gm_id = 0
wm_id = 1
csf_id = 2
nt = 3 # number of tissue types

## T1
vol_t1 = np.zeros([ns_t1,nt])
for i in range(ns_t1):
    vol_t1[i,:] = np.genfromtxt(brainager_dir+'/'+sub_t1[i]+'/t1/'+sub_t1[i]+'_t1_tissue_volumes.csv',skip_header=1,delimiter=',')[1:4]

## T1-FOV(EPI)
vol_t1_fov = np.zeros([ns_t1,nt])
for i in range(ns_t1):
    vol_t1_fov[i,:] = np.genfromtxt(brainager_dir+'/'+sub_t1[i]+'/t1_epifov/'+sub_t1[i]+'_t1_epifov_tissue_volumes.csv',skip_header=1,delimiter=',')[1:4]

## EPImix
vol_epi = np.zeros([ns_epi,nt])
for i in range(ns_epi):
    vol_epi[i,:] = np.genfromtxt(brainager_dir+'/'+sub_epi[i]+'/epimix/'+sub_epi[i]+'_epimix_tissue_volumes.csv',skip_header=1,delimiter=',')[1:4]

# %% Global Volume comparison - Full FoV

## GM
# limits and ticks
ax_lim = [0.4, 1.0]
ax_ticks = np.arange(0.4, 1.2, 0.2)
# plot
plt.figure(); plt.axis('square')                        # plot set-up
plt.plot(ax_lim, ax_lim, 'k--')                         # plot dashed identity line
sb.regplot(vol_t1[:,gm_id], vol_epi[ids_t1_subset,gm_id], ci=95, scatter_kws={"color": "orangered"}, line_kws={"color": "black"})  # line of best fit + CI
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs)     # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs)
plt.xlabel('T$_1$-w vol. (litres)',fontsize=lbs+2)       # labels
plt.ylabel('EPImix T$_1$-w vol. (litres)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/glob_vol_gm.svg',bbox_inches='tight')

## WM
# limits and ticks
ax_lim = [0.3,0.7]
ax_ticks = np.arange(0.3, 0.8, 0.1)
# plot
plt.figure(); plt.axis('square')                        # plot set-up
plt.plot(ax_lim, ax_lim, 'k--')                         # plot dashed identity line
sb.regplot(vol_t1[:,wm_id], vol_epi[ids_t1_subset,wm_id], ci=95, scatter_kws={"color": "dimgray"}, line_kws={"color": "black"})  # line of best fit + CI
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs)     # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs)
plt.xlabel('T$_1$-w  vol. (litres)',fontsize=lbs+2)       # labels
plt.ylabel('EPImix T$_1$-w vol. (litres)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/glob_vol_wm.svg',bbox_inches='tight')

## CSF
# limits and ticks
ax_lim = [0.0, 0.5]
ax_ticks = np.arange(0.0, 0.6, 0.1)
# plot
plt.figure(); plt.axis('square')                        # plot set-up
plt.plot(ax_lim, ax_lim, 'k--')                         # plot dashed identity line
sb.regplot(vol_t1[:,csf_id], vol_epi[ids_t1_subset,csf_id], ci=95, scatter_kws={"color": "fuchsia"}, line_kws={"color": "black"})  # line of best fit + CI
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs)     # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs)
plt.xlabel('T$_1$-w vol. (litres)',fontsize=lbs+2)       # labels
plt.ylabel('EPImix T$_1$-w vol. (litres)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/glob_vol_csf.svg',bbox_inches='tight')

# %% statistics for Global Volume

## Spearman rho
stats.spearmanr(vol_t1[:,gm_id], vol_epi[ids_t1_subset,gm_id])          # GM
stats.spearmanr(vol_t1[:,wm_id], vol_epi[ids_t1_subset,wm_id])          # WM
stats.spearmanr(vol_t1[:,csf_id], vol_epi[ids_t1_subset,csf_id])        # CSF

## Pearson's r-squared
stats.pearsonr(vol_t1[:,gm_id], vol_epi[ids_t1_subset,gm_id])[0]**2     # GM
stats.pearsonr(vol_t1[:,wm_id], vol_epi[ids_t1_subset,wm_id])[0]**2     # WM
stats.pearsonr(vol_t1[:,csf_id], vol_epi[ids_t1_subset,csf_id])[0]**2   # CSF
    
# %% Global Volume comparison - Reduced FoV

## GM
# limits and ticks
ax_lim = [0.4, 1.0]
ax_ticks = np.arange(0.4, 1.2, 0.2)
# plot
plt.figure(); plt.axis('square')                        # plot set-up
plt.plot(ax_lim, ax_lim, 'k--')                         # plot dashed identity line
sb.regplot(vol_t1_fov[:,gm_id], vol_epi[ids_t1_subset,gm_id], ci=95, scatter_kws={"color": "orangered"}, line_kws={"color": "black"})  # line of best fit + CI
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs)     # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs)
plt.xlabel('T$_1$-w vol. (litres)',fontsize=lbs+2)       # labels
plt.ylabel('EPImix T$_1$-w vol. (litres)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/glob_vol_gm_fov.svg',bbox_inches='tight')

## WM
# limits and ticks
ax_lim = [0.3,0.7]
ax_ticks = np.arange(0.3, 0.8, 0.1)
# plot
plt.figure(); plt.axis('square')                        # plot set-up
plt.plot(ax_lim, ax_lim, 'k--')                         # plot dashed identity line
sb.regplot(vol_t1_fov[:,wm_id], vol_epi[ids_t1_subset,wm_id], ci=95, scatter_kws={"color": "dimgray"}, line_kws={"color": "black"})  # line of best fit + CI
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs)     # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs)
plt.xlabel('T$_1$-w vol. (litres)',fontsize=lbs+2)       # labels
plt.ylabel('EPImix T$_1$-w vol. (litres)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/glob_vol_wm_fov.svg',bbox_inches='tight')

## CSF
# limits and ticks
ax_lim = [0.0, 0.5]
ax_ticks = np.arange(0.0, 0.6, 0.1)
# plot
plt.figure(); plt.axis('square')                        # plot set-up
plt.plot(ax_lim, ax_lim, 'k--')                         # plot dashed identity line
sb.regplot(vol_t1_fov[:,csf_id], vol_epi[ids_t1_subset,csf_id], ci=95, scatter_kws={"color": "fuchsia"}, line_kws={"color": "black"})  # line of best fit + CI
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs)     # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs)
plt.xlabel('T$_1$-w vol. (litres)',fontsize=lbs+2)       # labels
plt.ylabel('EPImix T$_1$-w vol. (litres)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/glob_vol_csf_fov.svg',bbox_inches='tight')

# %% statistics for Global Volume - Reduced FoV

## Spearman rho
stats.spearmanr(vol_t1_fov[:,gm_id], vol_epi[ids_t1_subset,gm_id])          # GM
stats.spearmanr(vol_t1_fov[:,wm_id], vol_epi[ids_t1_subset,wm_id])          # WM
stats.spearmanr(vol_t1_fov[:,csf_id], vol_epi[ids_t1_subset,csf_id])        # CSF

## Pearson's r-squared
stats.pearsonr(vol_t1_fov[:,gm_id], vol_epi[ids_t1_subset,gm_id])[0]**2     # GM
stats.pearsonr(vol_t1_fov[:,wm_id], vol_epi[ids_t1_subset,wm_id])[0]**2     # WM
stats.pearsonr(vol_t1_fov[:,csf_id], vol_epi[ids_t1_subset,csf_id])[0]**2   # CSF
    
# %% Grey Matter volume as a function of (chronological) age

# plot set-up
age_x_ticks = np.arange(10, 75, 10)
age_y_ticks = np.arange(0.4, 1.4, 0.2)
age_x_lim = [13, 65]
age_y_lim = [0.4, 1.05]

# T1
plt.figure()
plt.xticks(age_x_ticks,fontsize=axs); plt.xlim(age_x_lim); 
plt.yticks(age_y_ticks,fontsize=axs); plt.ylim(age_y_lim);
sb.regplot(age[ids_t1_subset], vol_t1[:,gm_id], ci=95, scatter_kws={"color": "indigo"}, line_kws={"color": "black"}) 
plt.xlabel("Chronological Age (years)",fontsize=lbs+2)
plt.ylabel('GM volume (litres)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/gm_vs_age_t1.svg',bbox_inches='tight')

# EPImix
plt.figure()
plt.xticks(age_x_ticks,fontsize=axs); plt.xlim(age_x_lim); 
plt.yticks(age_y_ticks,fontsize=axs); plt.ylim(age_y_lim);
sb.regplot(age[ids_t1_subset], vol_epi[ids_t1_subset,gm_id], ci=95, scatter_kws={"color": "royalblue"}, line_kws={"color": "black"}) 
plt.xlabel("Chronological Age (years)",fontsize=lbs+2)
plt.ylabel('GM volume (litres)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/gm_vs_age_epi.svg',bbox_inches='tight')

# T1 FoV
plt.figure()
plt.xticks(age_x_ticks,fontsize=axs); plt.xlim(age_x_lim); 
plt.yticks(age_y_ticks,fontsize=axs); plt.ylim(age_y_lim);
sb.regplot(age[ids_t1_subset], vol_t1_fov[:,gm_id], ci=95, scatter_kws={"color": "mediumpurple"}, line_kws={"color": "black"}) 
plt.xlabel("Chronological Age (years)",fontsize=lbs+2)
plt.ylabel('GM volume (litres)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/gm_vs_age_t1_fov.svg',bbox_inches='tight')

# EPImix - all participants
plt.figure()
plt.xticks(age_x_ticks,fontsize=axs); plt.xlim(age_x_lim); 
plt.yticks(age_y_ticks,fontsize=axs); plt.ylim(age_y_lim);
sb.regplot(age, vol_epi[:,gm_id], ci=95, scatter_kws={"color": "royalblue"}, line_kws={"color": "black"}) 
plt.xlabel("Chronological Age (years)",fontsize=lbs+2)
plt.ylabel('GM volume (litres)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/gm_vs_age_epi_ALL.svg',bbox_inches='tight')

# %% statistics for Grey Matter volume VS (chronological) age

## Spearman rho
stats.spearmanr(age[ids_t1_subset], vol_epi[ids_t1_subset,gm_id])       # EPImix
stats.spearmanr(age[ids_t1_subset], vol_t1[:,gm_id])                    # T1
stats.spearmanr(age[ids_t1_subset], vol_t1_fov[:,gm_id])                # T1 FoV
stats.spearmanr(age, vol_epi[:,gm_id])                                  # EPImix - ALL

## Pearson's r-squared
stats.pearsonr(age[ids_t1_subset], vol_epi[ids_t1_subset,gm_id])[0]**2  # EPImix
stats.pearsonr(age[ids_t1_subset], vol_t1[:,gm_id])[0]**2               # T1
stats.pearsonr(age[ids_t1_subset], vol_t1_fov[:,gm_id])[0]**2           # T1 FoV
stats.pearsonr(age, vol_epi[:,gm_id])[0]**2                             # EPImix - ALL

# %%

"""

Tissue volumes - Voxel-wise

"""

# %% Load voxel-wise data

## example nii volume - for number of voxels, shape and affine matrix
ex_nii = nib.load(brainager_dir+'/ADAPTA01/epimix/smwc1ADAPTA01_epimix.nii')    # load image

# extract parameters from example volume
nvox = np.array(ex_nii.dataobj).size                                        # number of voxels
nii_shape = np.array(ex_nii.dataobj).shape                                  # image dimensions
nii_affine = ex_nii.affine                                                  # affine matrix

# load EPImix data - for all participants (N = 94) 
epi_gm = np.zeros([nvox,ns_epi])
epi_wm = np.zeros([nvox,ns_epi])
epi_csf = np.zeros([nvox,ns_epi])
for i in range(ns_epi):
    print(sub_epi[i])
    epi_gm[:,i] = np.array(nib.load(brainager_dir+'/'+sub_epi[i]+'/epimix/smwc1'+sub_epi[i]+'_epimix.nii').dataobj).flatten()
    epi_wm[:,i] = np.array(nib.load(brainager_dir+'/'+sub_epi[i]+'/epimix/smwc2'+sub_epi[i]+'_epimix.nii').dataobj).flatten()
    epi_csf[:,i] = np.array(nib.load(brainager_dir+'/'+sub_epi[i]+'/epimix/smwc3'+sub_epi[i]+'_epimix.nii').dataobj).flatten()

# load T1 data - for participants with T1 data (N = 64)
t1_gm = np.zeros([nvox,ns_t1])
t1_wm = np.zeros([nvox,ns_t1])
t1_csf = np.zeros([nvox,ns_t1])
for i in range(ns_t1):
    print(sub_t1[i])
    t1_gm[:,i] = np.array(nib.load(brainager_dir+'/'+sub_t1[i]+'/t1/smwc1'+sub_t1[i]+'_t1.nii').dataobj).flatten()
    t1_wm[:,i] = np.array(nib.load(brainager_dir+'/'+sub_t1[i]+'/t1/smwc2'+sub_t1[i]+'_t1.nii').dataobj).flatten()
    t1_csf[:,i] = np.array(nib.load(brainager_dir+'/'+sub_t1[i]+'/t1/smwc3'+sub_t1[i]+'_t1.nii').dataobj).flatten()

# %% EPImix FoV masks
# create masks of voxels with at least 0.1% tissue in at least 95% participants (in EPImix scans)

thr_vol = 0.001 # threshold for minimum percentage of tissue volume
thr_part = 0.95 # threshold for minimum proportion of participants

# proportion of non-zero voxels (across participants)
overlap_gm = np.sum((epi_gm>thr_vol)*1,1)/ns_epi
overlap_wm = np.sum((epi_wm>thr_vol)*1,1)/ns_epi
overlap_csf = np.sum((epi_csf>thr_vol)*1,1)/ns_epi

# mask of voxels where "thr_part" (proportion) of scans are covered
fov_mask_gm = (overlap_gm>thr_part)*1
fov_mask_wm = (overlap_wm>thr_part)*1
fov_mask_csf = (overlap_csf>thr_part)*1

# indices of non-zero voxels
fov_gm_ind = np.where(fov_mask_gm==1)[0]
fov_wm_ind = np.where(fov_mask_wm==1)[0]
fov_csf_ind = np.where(fov_mask_csf==1)[0] 

# %% Mask visualisation
# the variables created below are purely for visualisation purposes (as opposed to analysis)

# masks for visualisation of non-zero voxels only; i.e. voxels with at least one participant having non-zero volume (i.e. volume >1e-6)
nz_mask_gm = ((np.sum((epi_gm>1e-6)*1,1)/ns_epi)>0)*1
nz_mask_wm = ((np.sum((epi_wm>1e-6)*1,1)/ns_epi)>0)*1
nz_mask_csf = ((np.sum((epi_csf>1e-6)*1,1)/ns_epi)>0)*1

# fov values for voxels within mask
overlap_fov_gm = fov_mask_gm*((nz_mask_gm!=0)*1)
overlap_fov_wm = fov_mask_wm*((nz_mask_wm!=0)*1)
overlap_fov_csf = fov_mask_csf*((nz_mask_csf!=0)*1)

### plot overlap - parameters
c_lim = (0,1)        # colorbar limit
cut_crd = (30, 0)    # FoV coordinates

# GM
# mask only
plot_nl_image_masked(overlap_gm, nz_mask_gm, nii_shape, ex_nii.affine, cmap='plasma_r', clim=c_lim, cut_coords=cut_crd, draw_cross=True,black_bg=False,display_mode='yx')
if save_fig: plt.savefig(plot_dir+'/overlap_gm.png',dpi=500,bbox_inches='tight')
# mask && FoV only
plot_nl_image_masked(overlap_gm, overlap_fov_gm, nii_shape, ex_nii.affine, cmap='plasma_r', clim=c_lim, cut_coords=cut_crd, draw_cross=True,black_bg=False,display_mode='yx')
if save_fig: plt.savefig(plot_dir+'/overlap_gm_fov.png',dpi=500,bbox_inches='tight')

# WM
# mask only
plot_nl_image_masked(overlap_wm, nz_mask_wm, nii_shape, ex_nii.affine, cmap='plasma_r', clim=c_lim, cut_coords=cut_crd, draw_cross=True,black_bg=False,display_mode='yx')
if save_fig: plt.savefig(plot_dir+'/overlap_wm.png',dpi=500,bbox_inches='tight')
# mask && FoV only
plot_nl_image_masked(overlap_wm, overlap_fov_wm, nii_shape, ex_nii.affine, cmap='plasma_r', clim=c_lim, cut_coords=cut_crd, draw_cross=True,black_bg=False,display_mode='yx')
if save_fig: plt.savefig(plot_dir+'/overlap_wm_fov.png',dpi=500,bbox_inches='tight')

# CSF
# mask only
plot_nl_image_masked(overlap_csf, nz_mask_csf, nii_shape, ex_nii.affine, cmap='plasma_r', clim=c_lim, cut_coords=cut_crd, draw_cross=True,black_bg=False,display_mode='yx')
if save_fig: plt.savefig(plot_dir+'/overlap_csf.png',dpi=500,bbox_inches='tight')
# mask && FoV only
plot_nl_image_masked(overlap_csf, overlap_fov_csf, nii_shape, ex_nii.affine, cmap='plasma_r', clim=c_lim, cut_coords=cut_crd, draw_cross=True,black_bg=False,display_mode='yx')
if save_fig: plt.savefig(plot_dir+'/overlap_csf_fov.png',dpi=500,bbox_inches='tight')

# %% Calculate correlation within FoV mask
## i.e. calculate correlation only within voxels belonging to FoV for the given tissue
## !!! TAKES A LONG TIME TO RUN !!!

# GM
rho_gm = np.zeros(nvox)
for i in range(len(fov_gm_ind)):
    if i % 10000 == 0: print(i) # track progress
    rho_gm[fov_gm_ind[i]] = stats.spearmanr(epi_gm[fov_gm_ind[i],ids_t1_subset],t1_gm[fov_gm_ind[i],:])[0] # "ids_t1_subset" used to index EPImix scans of participants who also have T1 data
    
# WM
rho_wm = np.zeros(nvox)
for i in range(len(fov_wm_ind)):
    if i % 10000 == 0: print(i) # track progress
    rho_wm[fov_wm_ind[i]] = stats.spearmanr(epi_wm[fov_wm_ind[i],ids_t1_subset],t1_wm[fov_wm_ind[i],:])[0] # "ids_t1_subset" used to index EPImix scans of participants who also have T1 data
    
# CSF
rho_csf = np.zeros(nvox)
for i in range(len(fov_csf_ind)):
    if i % 10000 == 0: print(i) # track progress
    rho_csf[fov_csf_ind[i]] = stats.spearmanr(epi_csf[fov_csf_ind[i],ids_t1_subset],t1_csf[fov_csf_ind[i],:])[0] # "ids_t1_subset" used to index EPImix scans of participants who also have T1 data

# %% visualise results

## plot nii using customised nilearn function
# GM
plot_nl_image_masked(rho_gm, overlap_fov_gm, nii_shape, ex_nii.affine, cmap='PuOr', clim=(-1,1), cut_coords=np.arange(-20,60,15), black_bg=False,display_mode='z')
if save_fig: plt.savefig(plot_dir+'/voxelwise_rho_gm_axial.png',dpi=500,bbox_inches='tight')
# WM
plot_nl_image_masked(rho_wm, overlap_fov_wm, nii_shape, ex_nii.affine, cmap='PuOr', clim=(-1,1), cut_coords=np.arange(-20,60,15), black_bg=False,display_mode='z')
if save_fig: plt.savefig(plot_dir+'/voxelwise_rho_wm_axial.png',dpi=500,bbox_inches='tight')
# CSF
plot_nl_image_masked(rho_csf, overlap_fov_csf, nii_shape, ex_nii.affine, cmap='PuOr', clim=(-1,1), cut_coords=np.arange(-20,60,15), black_bg=False,display_mode='z')
if save_fig: plt.savefig(plot_dir+'/voxelwise_rho_csf_axial.png',dpi=500,bbox_inches='tight')

# # reshape correlation values into nifti images
# rho_gm_nii = nib.Nifti1Image(np.reshape(rho_gm, nii_shape), nii_affine)
# rho_wm_nii = nib.Nifti1Image(np.reshape(rho_wm, nii_shape), nii_affine)
# rho_csf_nii = nib.Nifti1Image(np.reshape(rho_csf, nii_shape), nii_affine)

# ## plot nii using nilearn
# # GM
# nl.plotting.plot_img(rho_gm_nii, display_mode='z', cut_coords=np.arange(-20,60,15), draw_cross=True, cmap='PuOr', vmin=-1, vmax = 1, colorbar= True)
# if save_fig: plt.savefig(plot_dir+'/voxelwise_rho_gm_axial_grey.png',dpi=500,bbox_inches='tight')
# # WM
# nl.plotting.plot_img(rho_wm_nii, display_mode='z', cut_coords=np.arange(-20,60,15), draw_cross=True, cmap='PuOr', vmin=-1, vmax = 1, colorbar= True)
# if save_fig: plt.savefig(plot_dir+'/voxelwise_rho_wm_axial_grey.png',dpi=500,bbox_inches='tight')
# # CSF
# nl.plotting.plot_img(rho_csf_nii, display_mode='z', cut_coords=np.arange(-20,60,15), draw_cross=True, cmap='PuOr', vmin=-1, vmax = 1, colorbar= True)
# if save_fig: plt.savefig(plot_dir+'/voxelwise_rho_csf_axial_grey.png',dpi=500,bbox_inches='tight')

# %% Kernel density plots of distribution of correlations

dens_x_ticks = np.arange(-0.4,1.2,0.2)
dens_y_ticks = np.arange(0,3.5,0.5)

# GM
plt.figure()
sb.kdeplot(rho_gm[fov_gm_ind], bw_method='scott',color='orangered', label="Grey Matter")
plt.xlabel(r"Spearman's $\rho$",fontsize=lbs+3,labelpad=10); plt.ylabel("Probability density",fontsize=lbs+3,labelpad=10)
plt.xticks(dens_x_ticks,fontsize=axs); plt.yticks(dens_y_ticks,fontsize=axs)
if save_fig: plt.savefig(plot_dir+'/kernel_density_rho_gm.svg',bbox_inches='tight') 

# WM
plt.figure()
sb.kdeplot(rho_wm[fov_wm_ind], bw_method='scott',color='dimgray', label="White Matter")
plt.xlabel(r"Spearman's $\rho$",fontsize=lbs+3,labelpad=10); plt.ylabel("Probability density",fontsize=lbs+3,labelpad=10)
plt.xticks(dens_x_ticks,fontsize=axs); plt.yticks(dens_y_ticks,fontsize=axs)
if save_fig: plt.savefig(plot_dir+'/kernel_density_rho_wm.svg',bbox_inches='tight') 

# CSF
plt.figure()
sb.kdeplot(rho_csf[fov_csf_ind], bw_method='scott',color='fuchsia', label="CSF")
plt.xlabel(r"Spearman's $\rho$",fontsize=lbs+3,labelpad=10); plt.ylabel("Probability density",fontsize=lbs+3,labelpad=10)
plt.xticks(dens_x_ticks,fontsize=axs); plt.yticks(dens_y_ticks,fontsize=axs)
if save_fig: plt.savefig(plot_dir+'/kernel_density_rho_csf.svg',bbox_inches='tight') 

# # combined
# vol_all = pd.DataFrame({'vol': np.concatenate((rho_gm[fov_gm_ind],rho_wm[fov_wm_ind],rho_csf[fov_csf_ind])), 
#                         'tissue': np.concatenate((np.repeat('gm',len(fov_gm_ind)),np.repeat('wm',len(fov_wm_ind)),np.repeat('csf',len(fov_csf_ind))))})
# sb.kdeplot(data=vol_all, x="vol", hue="tissue")
# plt.xlabel("Spearman's rho")
# if save_fig: plt.savefig(plot_dir+'/kernel_density_all.svg',bbox_inches='tight') 

# %% median and quartile values for each tissue class

# including formatting
str(round(np.median(rho_gm[fov_gm_ind]),2)) +' ['+ str(round(np.percentile(rho_gm[fov_gm_ind],25),2)) +','+ str(round(np.percentile(rho_gm[fov_gm_ind],75),2)) +']'
str(round(np.median(rho_wm[fov_wm_ind]),2)) +' ['+ str(round(np.percentile(rho_wm[fov_wm_ind],25),2)) +','+ str(round(np.percentile(rho_wm[fov_wm_ind],75),2)) +']'
str(round(np.median(rho_csf[fov_csf_ind]),2)) +' ['+ str(round(np.percentile(rho_csf[fov_csf_ind],25),2)) +','+ str(round(np.percentile(rho_csf[fov_csf_ind],75),2)) +']'

# %%

"""

Brain age

"""

# %% read in predicted brain age from csv files 
# csv files contain predicted brain age as [file_name, age, lower CI, upper CI] -> brain age is 1st value

# brain age for T1 scans 
brain_age_t1 = np.zeros(ns_t1)
for i in range(ns_t1):
    brain_age_t1[i] = np.genfromtxt(brainager_dir+'/'+sub_t1[i]+'/t1/pred_brain_age.csv',skip_header=1,delimiter=',')[1]
    
# brain age for T1 scans with reduced FoV
brain_age_t1_fov = np.zeros(ns_t1)
for i in range(ns_t1):
    brain_age_t1_fov[i] = np.genfromtxt(brainager_dir+'/'+sub_t1[i]+'/t1_epifov/pred_brain_age.csv',skip_header=1,delimiter=',')[1]

# brain age from EPImix scans
brain_age_epi = np.zeros(ns_epi)
for i in range(ns_epi):
    brain_age_epi[i] = np.genfromtxt(brainager_dir+'/'+sub_epi[i]+'/epimix/pred_brain_age.csv',skip_header=1,delimiter=',')[1]

# %% plots of predicted age against chronological age

# hardcode axis limits and ticks (to make plot nice)
ax_lim = [10,65]
ax_ticks = np.arange(20, 65, 10)

## EPImix - all participants
plt.figure(); plt.axis('square')                        # plot set-up
plt.plot(ax_lim, ax_lim, 'k--')                         # plot dashed identity line
sb.regplot(age[ids_t1_subset], brain_age_epi[ids_t1_subset], ci=95, scatter_kws={"color": "royalblue"}, line_kws={"color": "black"})  # line of best fit + CI
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs)     # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs)
plt.xlabel('Chronological Age (y)',fontsize=lbs+2)    # labels
plt.ylabel('Predicted Age (y)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/chron_pred_epi.svg',bbox_inches='tight')

## T1
plt.figure(); plt.axis('square')                        # plot set-up
plt.plot(ax_lim, ax_lim, 'k--')                         # plot dashed identity line
sb.regplot(age[ids_t1_subset],brain_age_t1, ci=95, scatter_kws={"color": "indigo"}, line_kws={"color": "black"})  # line of best fit + CI
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs)     # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs)
plt.xlabel('Chronological Age (y)',fontsize=lbs+2)    # labels
plt.ylabel('Predicted Age (y)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/chron_pred_t1.svg',bbox_inches='tight')

## T1 with reduced FoV
plt.figure(); plt.axis('square')                        # plot set-up
plt.plot(ax_lim, ax_lim, 'k--')                         # plot dashed identity line
sb.regplot(age[ids_t1_subset],brain_age_t1_fov, ci=95, scatter_kws={"color": "mediumpurple"}, line_kws={"color": "black"})  # line of best fit + CI
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs)     # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs)
plt.xlabel('Chronological Age (y)',fontsize=lbs+2)    # labels
plt.ylabel('Predicted Age (y)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/chron_pred_t1_fov.svg',bbox_inches='tight')

## EPImix - all participants
plt.figure(); plt.axis('square')                        # plot set-up
plt.plot(ax_lim, ax_lim, 'k--')                         # plot dashed identity line
sb.regplot(age, brain_age_epi, ci=95, scatter_kws={"color": "royalblue"}, line_kws={"color": "black"})  # line of best fit + CI
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs)     # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs)
plt.xlabel('Chronological Age (y)',fontsize=lbs+2)    # labels
plt.ylabel('Predicted Age (y)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/chron_pred_epi_ALL.svg',bbox_inches='tight')

# %% statistics for predicted age VS chronological age

## Median Absolute Error 
mae_epi = median_absolute_error(age[ids_t1_subset], brain_age_epi[ids_t1_subset])   # EPImix
mae_t1 = median_absolute_error(age[ids_t1_subset], brain_age_t1)                    # T1
mae_t1_fov = median_absolute_error(age[ids_t1_subset], brain_age_t1_fov)            # T1 FoV
#median_absolute_error(age,brain_age_epi)                               # EPImix - ALL

## Spearman rho
stats.spearmanr(age[ids_t1_subset], brain_age_epi[ids_t1_subset])       # EPImix
stats.spearmanr(age[ids_t1_subset], brain_age_t1)                       # T1
stats.spearmanr(age[ids_t1_subset], brain_age_t1_fov)                   # T1 FoV
stats.spearmanr(age,brain_age_epi)                                      # EPImix - ALL

## Pearson's r-squared
stats.pearsonr(age[ids_t1_subset], brain_age_epi[ids_t1_subset])[0]**2  # EPImix
stats.pearsonr(age[ids_t1_subset], brain_age_t1)[0]**2                  # T1
stats.pearsonr(age[ids_t1_subset], brain_age_t1_fov)[0]**2              # T1 FoV
stats.pearsonr(age,brain_age_epi)[0]**2                                 # EPImix - ALL

# %% plots of predicted ages 

## T1 VS EPImix
plt.figure(); plt.axis('square')                        # plot set-up
plt.plot(ax_lim, ax_lim, 'k--')                         # plot dashed identity line
sb.regplot(brain_age_t1, brain_age_epi[ids_t1_subset], ci=95, scatter_kws={"color": "darkred"}, line_kws={"color": "black"})  # line of best fit + CI
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs)     # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs)
plt.xlabel('T$_1$-w Pred. Age (y)',fontsize=lbs+2)
plt.ylabel('EPImix T$_1$-w Pred. Age (y)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/pred_age_t1_vs_epi.svg',bbox_inches='tight')

## T1 with reduced FoV VS EPImix
plt.figure(); plt.axis('square')                        # plot set-up
plt.plot([26, 48], [26, 48], 'k--')                     # plot dashed identity line
sb.regplot(brain_age_t1_fov, brain_age_epi[ids_t1_subset], ci=95, scatter_kws={"color": "darkred"}, line_kws={"color": "black"})  # line of best fit + CI
plt.xlim([26, 48]); plt.xticks(np.arange(30, 48, 4),fontsize=axs)   # axis limits and ticks
plt.ylim([26, 48]); plt.yticks(np.arange(30, 48, 4),fontsize=axs)
plt.xlabel('T$_1$-w FoV$_{EPI}$ Pred. Age (y)',fontsize=lbs+2)
plt.ylabel('EPImix T$_1$-w Pred. Age (y)',fontsize=lbs+2)
if save_fig: plt.savefig(plot_dir+'/pred_age_t1_fov_vs_epi.svg',bbox_inches='tight')

# %% statistics for predicted ages

## Spearman rho
stats.spearmanr(brain_age_t1, brain_age_epi[ids_t1_subset])             # T1 VS EPImix
stats.spearmanr(brain_age_t1_fov, brain_age_epi[ids_t1_subset])         # T1 FoV VS EPImix

## Pearson's r-squared
stats.pearsonr(brain_age_t1, brain_age_epi[ids_t1_subset])[0]**2        # T1 VS EPImix
stats.pearsonr(brain_age_t1_fov, brain_age_epi[ids_t1_subset])[0]**2    # T1 FoV VS EPImix

#%% leave-one-out cross-validation to adjust predicted brain age estimate

linreg = linear_model.LinearRegression()

linreg_pred_loo = np.zeros([ns_t1])
for i in range(ns_t1): # loop over all participants
    # extract indices for current iteration of loop
    train_i_loo = np.setdiff1d(np.arange(0,ns_t1),i)   # training indices (all except one participant)
    test_i_loo = i                                      # testing (left-out) index
    # fitting and prediction
    linreg.fit(brain_age_epi[ids_t1_subset][train_i_loo].reshape(-1,1),brain_age_t1[train_i_loo].reshape(-1,1))    # fit linear model
    linreg_pred_loo[test_i_loo] = linreg.predict(brain_age_epi[ids_t1_subset][test_i_loo].reshape(-1,1))[:,0]       # predict "adjusted" value

# adjusted MAE
linreg_mae_loo = median_absolute_error(age[ids_t1_subset],linreg_pred_loo)

# %% comparison with null model
# null model = "worst possible brain age estimate", based on the assumption that the same age would be predicted for each participant,
# equal to the average age of the brainageR omdel training set (40.6 years)

null_age = np.repeat(40.6,ns_t1)

## null model Median Absolute Error 
null_mae = median_absolute_error(age[ids_t1_subset], null_age)

# null ratio - ratio of actual MAE to null MAE
null_mae/mae_epi    # EPImix
null_mae/mae_t1     # T1
null_mae/mae_t1_fov # T1 FoV

# %%

"""

Test-retest reliability

"""

# %% test-retest subject IDs

# read in ID's of participants with epimixretest_scans
sub_epi_rt = np.genfromtxt(demog_dir+'/brainager_epimix_rt_ids.txt',dtype='str')
ns_epi_rt = len(sub_epi_rt) # number of participants

# extract numerical IDs of position of the subset of subjects with EPImix retest scans relative to all subjects
ids_epi_rt_subset = np.zeros(ns_epi_rt,dtype=int)
for i in range(ns_epi_rt):
    ids_epi_rt_subset[i] = np.where(sub_epi_rt[i]==sub_epi)[0]

# %% test-retest data

# brain age
brain_age_epi_rt = np.zeros(ns_epi_rt)
for i in range(ns_epi_rt):
    brain_age_epi_rt[i] = np.genfromtxt(brainager_dir+'/'+sub_epi_rt[i]+'/epimix_RT/pred_brain_age.csv',skip_header=1,delimiter=',')[1]

# global volume
vol_epi_rt = np.zeros([ns_epi_rt,nt]) # nt = number of tissue types
for i in range(ns_epi_rt):
    vol_epi_rt[i,:] = np.genfromtxt(brainager_dir+'/'+sub_epi_rt[i]+'/epimix_RT/'+sub_epi_rt[i]+'_epimix_RT_tissue_volumes.csv',skip_header=1,delimiter=',')[1:4]

# voxel-wise volume
epi_rt_gm = np.zeros([nvox,ns_epi_rt])
epi_rt_wm = np.zeros([nvox,ns_epi_rt])
epi_rt_csf = np.zeros([nvox,ns_epi_rt])
for i in range(ns_epi_rt):
    print(sub_epi_rt[i])
    epi_rt_gm[:,i] = np.array(nib.load(brainager_dir+'/'+sub_epi_rt[i]+'/epimix_RT/smwc1'+sub_epi_rt[i]+'_epimix_RT.nii').dataobj).flatten()
    epi_rt_wm[:,i] = np.array(nib.load(brainager_dir+'/'+sub_epi_rt[i]+'/epimix_RT/smwc2'+sub_epi_rt[i]+'_epimix_RT.nii').dataobj).flatten()
    epi_rt_csf[:,i] = np.array(nib.load(brainager_dir+'/'+sub_epi_rt[i]+'/epimix_RT/smwc3'+sub_epi_rt[i]+'_epimix_RT.nii').dataobj).flatten()

# %% ICC3 for tissue volumes (global)

# # ids of different tissues in volume variables
# gm_id = 0
# wm_id = 1
# csf_id = 2

# GM
icc_gm = pg.intraclass_corr(data=pd.DataFrame({
    'subject':np.concatenate((np.arange(0,ns_epi_rt),np.arange(0,ns_epi_rt))),
    'scan':np.concatenate((np.repeat('T',ns_epi_rt),np.repeat('RT',ns_epi_rt))),
    'score':np.concatenate((vol_epi[ids_epi_rt_subset,gm_id],vol_epi_rt[:,gm_id]))}),
    targets='subject', raters='scan',ratings='score')
icc_gm['ICC'][2]    # ICC3
icc_gm['CI95%'][2]  # 95% CI

# WM
icc_wm = pg.intraclass_corr(data=pd.DataFrame({
    'subject':np.concatenate((np.arange(0,ns_epi_rt),np.arange(0,ns_epi_rt))),
    'scan':np.concatenate((np.repeat('T',ns_epi_rt),np.repeat('RT',ns_epi_rt))),
    'score':np.concatenate((vol_epi[ids_epi_rt_subset,wm_id],vol_epi_rt[:,wm_id]))}),
    targets='subject', raters='scan',ratings='score')
icc_wm['ICC'][2]    # ICC3
icc_wm['CI95%'][2]  # 95% CI

# CSF
icc_csf = pg.intraclass_corr(data=pd.DataFrame({
    'subject':np.concatenate((np.arange(0,ns_epi_rt),np.arange(0,ns_epi_rt))),
    'scan':np.concatenate((np.repeat('T',ns_epi_rt),np.repeat('RT',ns_epi_rt))),
    'score':np.concatenate((vol_epi[ids_epi_rt_subset,csf_id],vol_epi_rt[:,csf_id]))}),
    targets='subject', raters='scan',ratings='score')
icc_csf['ICC'][2]   # ICC3
icc_csf['CI95%'][2] # 95% CI

# %% ICC3 for tissue volumes (local)
## i.e. calculate ICC only within voxels belonging to FoV for the given tissue
## !!! TAKES A LONG TIME TO RUN !!!

#import time
#t0 = time.time()
# GM
icc_vox_gm = np.zeros(nvox) # voxelwise - within "brain" mask only
for i in range(len(fov_gm_ind)):
    if i % 10000 == 0: print(i)
    #@vectorize(["float32(float32, float32)"], target='cuda')
    icc_vox_gm[fov_gm_ind[i]] = pg.intraclass_corr(data=pd.DataFrame({
            'subject':np.concatenate((np.arange(0,ns_epi_rt),np.arange(0,ns_epi_rt))),
            'scan':np.concatenate((np.repeat('T',ns_epi_rt),np.repeat('RT',ns_epi_rt))),
            'score':np.concatenate((epi_gm[:,ids_epi_rt_subset],epi_rt_gm),axis=1)[fov_gm_ind[i],:]}),
        targets='subject', raters='scan',ratings='score')['ICC'][2]
np.save(home_dir+'/Python/icc_vox_gm.npy',icc_vox_gm)
#t1 = time.time()
#total = t1-t0

# WM
icc_vox_wm = np.zeros(nvox) # voxelwise - within "brain" mask only
for i in range(len(fov_wm_ind)):
    if i % 10000 == 0: print(i)
    icc_vox_wm[fov_wm_ind[i]] = pg.intraclass_corr(data=pd.DataFrame({
            'subject':np.concatenate((np.arange(0,ns_epi_rt),np.arange(0,ns_epi_rt))),
            'scan':np.concatenate((np.repeat('T',ns_epi_rt),np.repeat('RT',ns_epi_rt))),
            'score':np.concatenate((epi_wm[:,ids_epi_rt_subset],epi_rt_wm),axis=1)[fov_wm_ind[i],:]}),
        targets='subject', raters='scan',ratings='score')['ICC'][2]
np.save(home_dir+'/Python/icc_vox_wm.npy',icc_vox_wm)
    
# CSF
icc_vox_csf = np.zeros(nvox) # voxelwise - within "brain" mask only
for i in range(len(fov_csf_ind)):
    if i % 10000 == 0: print(i)
    icc_vox_csf[fov_csf_ind[i]] = pg.intraclass_corr(data=pd.DataFrame({
            'subject':np.concatenate((np.arange(0,ns_epi_rt),np.arange(0,ns_epi_rt))),
            'scan':np.concatenate((np.repeat('T',ns_epi_rt),np.repeat('RT',ns_epi_rt))),
            'score':np.concatenate((epi_csf[:,ids_epi_rt_subset],epi_rt_csf),axis=1)[fov_csf_ind[i],:]}),
        targets='subject', raters='scan',ratings='score')['ICC'][2]
np.save(home_dir+'/Python/icc_vox_csf.npy',icc_vox_csf)

# %% visualise voxel-wise ICC results

## plot nii using customised nilearn function
# GM
plot_nl_image_masked(icc_vox_gm, overlap_fov_gm, nii_shape, ex_nii.affine, cmap='PuOr', clim=(-1,1), cut_coords=np.arange(-20,60,15), black_bg=False,display_mode='z')
if save_fig: plt.savefig(plot_dir+'/voxelwise_icc_gm_axial.png',dpi=500,bbox_inches='tight')
# WM
plot_nl_image_masked(icc_vox_wm, overlap_fov_wm, nii_shape, ex_nii.affine, cmap='PuOr', clim=(-1,1), cut_coords=np.arange(-20,60,15), black_bg=False,display_mode='z')
if save_fig: plt.savefig(plot_dir+'/voxelwise_icc_wm_axial.png',dpi=500,bbox_inches='tight')
# CSF
plot_nl_image_masked(icc_vox_csf, overlap_fov_csf, nii_shape, ex_nii.affine, cmap='PuOr', clim=(-1,1), cut_coords=np.arange(-20,60,15), black_bg=False,display_mode='z')
if save_fig: plt.savefig(plot_dir+'/voxelwise_icc_csf_axial.png',dpi=500,bbox_inches='tight')

# # reshape correlation values into nifti images
# icc_vox_gm_nii = nib.Nifti1Image(np.reshape(icc_vox_gm, nii_shape), nii_affine)
# icc_vox_wm_nii = nib.Nifti1Image(np.reshape(icc_vox_wm, nii_shape), nii_affine)
# icc_vox_csf_nii = nib.Nifti1Image(np.reshape(icc_vox_csf, nii_shape), nii_affine)

# ## plot nii using nilearn
# # GM
# nl.plotting.plot_img(icc_vox_gm_nii, display_mode='z', cut_coords=np.arange(-20,60,15), draw_cross=True, cmap='PuOr', vmin=-1, vmax = 1, colorbar= True)
# if save_fig: plt.savefig(plot_dir+'/voxelwise_icc_gm_axial_grey.png',dpi=500,bbox_inches='tight')
# # WM
# nl.plotting.plot_img(icc_vox_wm_nii, display_mode='z', cut_coords=np.arange(-20,60,15), draw_cross=True, cmap='PuOr', vmin=-1, vmax = 1, colorbar= True)
# if save_fig: plt.savefig(plot_dir+'/voxelwise_icc_wm_axial_grey.png',dpi=500,bbox_inches='tight')
# # CSF
# nl.plotting.plot_img(icc_vox_csf_nii, display_mode='z', cut_coords=np.arange(-20,60,15), draw_cross=True, cmap='PuOr', vmin=-1, vmax = 1, colorbar= True)
# if save_fig: plt.savefig(plot_dir+'/voxelwise_icc_csf_axial_grey.png',dpi=500,bbox_inches='tight')

# %% Kernel density plots of distribution of ICC values

dens_x_ticks = np.arange(-1,1.1,0.5)
dens_y_ticks = np.arange(0,16,3)

# GM
plt.figure()
sb.kdeplot(icc_vox_gm[fov_gm_ind], bw_method='scott',color='orangered', label="Grey Matter")
plt.xlabel("ICC",fontsize=lbs+3,labelpad=10); plt.ylabel("Probability density",fontsize=lbs+3,labelpad=10)
plt.xticks(dens_x_ticks,fontsize=axs); plt.yticks(dens_y_ticks,fontsize=axs)
if save_fig: plt.savefig(plot_dir+'/kernel_density_icc_gm.svg',bbox_inches='tight') 

# WM
plt.figure()
sb.kdeplot(icc_vox_wm[fov_wm_ind], bw_method='scott',color='dimgray', label="White Matter")
plt.xlabel("ICC",fontsize=lbs+3,labelpad=10); plt.ylabel("Probability density",fontsize=lbs+3,labelpad=10)
plt.xticks(dens_x_ticks,fontsize=axs); plt.yticks(dens_y_ticks,fontsize=axs)
if save_fig: plt.savefig(plot_dir+'/kernel_density_icc_wm.svg',bbox_inches='tight') 

# CSF
plt.figure()
sb.kdeplot(icc_vox_csf[fov_csf_ind], bw_method='scott',color='fuchsia', label="CSF")
plt.xlabel("ICC",fontsize=lbs+3,labelpad=10); plt.ylabel("Probability density",fontsize=lbs+3,labelpad=10)
plt.xticks(dens_x_ticks,fontsize=axs); plt.yticks(dens_y_ticks,fontsize=axs)
if save_fig: plt.savefig(plot_dir+'/kernel_density_icc_csf.svg',bbox_inches='tight') 

# # combined
# vol_all = pd.DataFrame({'vol': np.concatenate((rho_gm[fov_gm_ind],rho_wm[fov_wm_ind],rho_csf[fov_csf_ind])), 
#                         'tissue': np.concatenate((np.repeat('gm',len(fov_gm_ind)),np.repeat('wm',len(fov_wm_ind)),np.repeat('csf',len(fov_csf_ind))))})
# sb.kdeplot(data=vol_all, x="vol", hue="tissue")
# plt.xlabel("Spearman's rho")
# if save_fig: plt.savefig(plot_dir+'/kernel_density_all.svg',bbox_inches='tight') 

# %% median and quartile values for each tissue class

# including formatting
str(round(np.median(icc_vox_gm[fov_gm_ind]),2)) +' ['+ str(round(np.percentile(icc_vox_gm[fov_gm_ind],25),2)) +','+ str(round(np.percentile(icc_vox_gm[fov_gm_ind],75),2)) +']'
str(round(np.median(icc_vox_wm[fov_wm_ind]),2)) +' ['+ str(round(np.percentile(icc_vox_wm[fov_wm_ind],25),2)) +','+ str(round(np.percentile(icc_vox_wm[fov_wm_ind],75),2)) +']'
str(round(np.median(icc_vox_csf[fov_csf_ind]),2)) +' ['+ str(round(np.percentile(icc_vox_csf[fov_csf_ind],25),2)) +','+ str(round(np.percentile(icc_vox_csf[fov_csf_ind],75),2)) +']'

# %% ICC3 for predicted brain age

icc_brain_age = pg.intraclass_corr(data=pd.DataFrame({
    'subject':np.concatenate((np.arange(0,ns_epi_rt),np.arange(0,ns_epi_rt))),
    'scan':np.concatenate((np.repeat('T',ns_epi_rt),np.repeat('RT',ns_epi_rt))),
    'score':np.concatenate((brain_age_epi[ids_epi_rt_subset],brain_age_epi_rt))}),
    targets='subject', raters='scan',ratings='score')
icc_brain_age['ICC'][2]    # ICC3
icc_brain_age['CI95%'][2]  # 95% CI
