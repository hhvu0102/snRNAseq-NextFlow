#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import tables
import anndata
from typing import Dict, Optional
import numpy as np
import scipy.sparse as sp
from scipy import io
import glob
import os
import upsetplot
from scipy.io import mmread
import csv

#### FUNCTIONS FROM CELLBENDER
def dict_from_h5(file: str) -> Dict[str, np.ndarray]:
    """Read in everything from an h5 file and put into a dictionary."""
    d = {}
    with tables.open_file(file) as f:
        # read in everything
        for array in f.walk_nodes("/", "Array"):
            d[array.name] = array.read()
    return d


def anndata_from_h5(file: str,
                    analyzed_barcodes_only: bool = True) -> 'anndata.AnnData':
    """Load an output h5 file into an AnnData object for downstream work.
    Args:
        file: The h5 file
        analyzed_barcodes_only: False to load all barcodes, so that the size of
            the AnnData object will match the size of the input raw count matrix.
            True to load a limited set of barcodes: only those analyzed by the
            algorithm. This allows relevant latent variables to be loaded
            properly into adata.obs and adata.obsm, rather than adata.uns.
    Returns:
        adata: The anndata object, populated with inferred latent variables
            and metadata.
    """

    d = dict_from_h5(file)
    X = sp.csc_matrix((d.pop('data'), d.pop('indices'), d.pop('indptr')),
                      shape=d.pop('shape')).transpose().tocsr()

    # check and see if we have barcode index annotations, and if the file is filtered
    barcode_key = [k for k in d.keys() if (('barcode' in k) and ('ind' in k))]
    if len(barcode_key) > 0:
        max_barcode_ind = d[barcode_key[0]].max()
        filtered_file = (max_barcode_ind >= X.shape[0])
    else:
        filtered_file = True

    if analyzed_barcodes_only:
        if filtered_file:
            # filtered file being read, so we don't need to subset
            print('Assuming we are loading a "filtered" file that contains only cells.')
            pass
        elif 'barcode_indices_for_latents' in d.keys():
            X = X[d['barcode_indices_for_latents'], :]
            d['barcodes'] = d['barcodes'][d['barcode_indices_for_latents']]
        elif 'barcodes_analyzed_inds' in d.keys():
            X = X[d['barcodes_analyzed_inds'], :]
            d['barcodes'] = d['barcodes'][d['barcodes_analyzed_inds']]
        else:
            print('Warning: analyzed_barcodes_only=True, but the key '
                  '"barcodes_analyzed_inds" or "barcode_indices_for_latents" '
                  'is missing from the h5 file. '
                  'Will output all barcodes, and proceed as if '
                  'analyzed_barcodes_only=False')

    # Construct the anndata object.
    adata = anndata.AnnData(X=X,
                            obs={'barcode': d.pop('barcodes').astype(str)},
                            var={'gene_name': (d.pop('gene_names') if 'gene_names' in d.keys()
                                               else d.pop('name')).astype(str)},
                            dtype=X.dtype)
    adata.obs.set_index('barcode', inplace=True)
    adata.var.set_index('gene_name', inplace=True)

    # For CellRanger v2 legacy format, "gene_ids" was called "genes"... rename this
    if 'genes' in d.keys():
        d['id'] = d.pop('genes')

    # For purely aesthetic purposes, rename "id" to "gene_id"
    if 'id' in d.keys():
        d['gene_id'] = d.pop('id')

    # If genomes are empty, try to guess them based on gene_id
    if 'genome' in d.keys():
        if np.array([s.decode() == '' for s in d['genome']]).all():
            if '_' in d['gene_id'][0].decode():
                print('Genome field blank, so attempting to guess genomes based on gene_id prefixes')
                d['genome'] = np.array([s.decode().split('_')[0] for s in d['gene_id']], dtype=str)

    # Add other information to the anndata object in the appropriate slot.
    _fill_adata_slots_automatically(adata, d)

    # Add a special additional field to .var if it exists.
    if 'features_analyzed_inds' in adata.uns.keys():
        adata.var['cellbender_analyzed'] = [True if (i in adata.uns['features_analyzed_inds'])
                                            else False for i in range(adata.shape[1])]

    if analyzed_barcodes_only:
        for col in adata.obs.columns[adata.obs.columns.str.startswith('barcodes_analyzed')
                                     | adata.obs.columns.str.startswith('barcode_indices')]:
            try:
                del adata.obs[col]
            except Exception:
                pass
    else:
        # Add a special additional field to .obs if all barcodes are included.
        if 'barcodes_analyzed_inds' in adata.uns.keys():
            adata.obs['cellbender_analyzed'] = [True if (i in adata.uns['barcodes_analyzed_inds'])
                                                else False for i in range(adata.shape[0])]

    return adata


def _fill_adata_slots_automatically(adata, d):
    """Add other information to the adata object in the appropriate slot."""

    # TODO: what about "features_analyzed_inds"?  If not all features are analyzed, does this work?

    for key, value in d.items():
        try:
            if value is None:
                continue
            value = np.asarray(value)
            if len(value.shape) == 0:
                adata.uns[key] = value
            elif value.shape[0] == adata.shape[0]:
                if (len(value.shape) < 2) or (value.shape[1] < 2):
                    adata.obs[key] = value
                else:
                    adata.obsm[key] = value
            elif value.shape[0] == adata.shape[1]:
                if value.dtype.name.startswith('bytes'):
                    adata.var[key] = value.astype(str)
                else:
                    adata.var[key] = value
            else:
                adata.uns[key] = value
        except Exception:
            print('Unable to load data into AnnData: ', key, value, type(value))


#### END FUNCTIONS FROM CELLBENDER

def cellbender_anndata_to_cell_probability(a):
    return a.obs.cell_probability


def cellbender_anndata_to_sparse_matrix(adata, min_cell_probability=0):
    barcodes = adata.obs[adata.obs.cell_probability>=min_cell_probability].index.to_list()
    features = adata.var.gene_id.to_list()
    matrix = adata[adata.obs.cell_probability>=min_cell_probability].X.transpose()
    return {'features': features, 'barcodes': barcodes, 'matrix': matrix}


def umi_count_after_decontamination(adata):
    x = cellbender_anndata_to_sparse_matrix(adata)
    return dict(zip(x['barcodes'], x['matrix'].sum(axis=0).tolist()[0]))


from skimage.filters import threshold_multiotsu
def estimate_threshold(x, classes=3, log_scale = True): #function to run Otsu 1D
    if log_scale == True: # do on logscale
        values = np.log10(x).values
    else:
        values = x.values
    values = values.reshape((len(values),1))
    thresholds = threshold_multiotsu(image=values, classes=classes, nbins=256)
    # convert back to linear scale
    if log_scale == True:
        thresholds = [pow(10, i) for i in thresholds]

    UMI_THRESHOLD = round(thresholds[classes - 2])
    return UMI_THRESHOLD


##### FUNCTIONS FOR MITO THRESHOLDS
from scipy.signal import find_peaks
import skimage as ski
from scipy import ndimage as ndi
from skimage import measure
import matplotlib.pyplot as plt

def guess_n_classes(metrics, mode = 'RNA'):
    modes = ['RNA', 'ATAC']
    if mode not in modes:
        raise ValueError("Invalid mode. Expected one of: %s" % modes)

    ## Step 1: guess lower bound to preclude
    if mode == "RNA":
        columns_to_check_missing = {'filter_rna_emptyDrops', 'filter_rna_min_umi',
                                    'rna_percent_mitochondrial', 'filter_pct_cellbender_removed'}
        if not columns_to_check_missing.issubset(metrics.columns):
            ValueError(f"Not all columns {columns_to_check_missing} exist in the DataFrame metrics.")

        data = metrics[(metrics.filter_rna_emptyDrops == True) & #assuming these columns exist in the metrics table already
                       (metrics.filter_rna_min_umi == True) &
                       (metrics.rna_percent_mitochondrial > 1) &
                       (metrics.rna_percent_mitochondrial < 50) &
                       (metrics.filter_pct_cellbender_removed == True)].rna_percent_mitochondrial.astype(float)
    else:
        columns_to_check_missing = {'filter_atac_min_hqaa', 'atac_percent_mitochondrial'}
        if not columns_to_check_missing.issubset(metrics.columns):
            ValueError(f"Not all columns {columns_to_check_missing} exist in the DataFrame metrics.")

        data = metrics[(metrics.filter_atac_min_hqaa == True) &
                       (metrics.atac_percent_mitochondrial > 1) &
                       (metrics.atac_percent_mitochondrial < 50)].atac_percent_mitochondrial.astype(float)

    values = np.log10(data).values
    values = values.reshape((len(values),1))
    thresholds = threshold_multiotsu(image=values, classes=4, nbins=256)
    # convert back to linear scale
    thresholds = [pow(10, i) for i in thresholds]

    # Step 2: get density and check number of distributions
    if mode == "RNA":
        data = np.log10(metrics[(metrics.filter_rna_emptyDrops == True) & #assuming these columns exist in the metrics table already
                               (metrics.filter_rna_min_umi == True) &
                               (metrics.rna_percent_mitochondrial > thresholds[0]) &
                               (metrics.rna_percent_mitochondrial < 50) &
                               (metrics.filter_pct_cellbender_removed == True)].rna_percent_mitochondrial.astype(float))
    else:
        data = metrics[(metrics.filter_atac_min_hqaa == True) &
                       (metrics.atac_percent_mitochondrial > thresholds[0]) &
                       (metrics.atac_percent_mitochondrial < 50)].atac_percent_mitochondrial.astype(float)

    kde = sns.kdeplot(data) # Generate KDE object from the data
    # The plotted data is stored in kde.lines[0].get_xdata() and .get_ydata()
    x = kde.lines[0].get_xdata()
    y = kde.lines[0].get_ydata()

    peaks, _ = find_peaks(y, prominence=abs(max(y) * 0.05))
    n_peaks = len(peaks)

    if mode == "RNA":
        print('Number of prominent cliff RNA %chrMT is {:,}'.format(n_peaks))
        rna_kde_df = pd.DataFrame({'x': x, 'density': y}) # Store in DataFrame for later plots
        plt.clf() # Clear the plot
        return n_peaks, rna_kde_df
    else:
        print('Number of prominent cliff ATAC %chrMT is {:,}'.format(n_peaks))
        atac_kde_df = pd.DataFrame({'x': x, 'density': y}) # Store in DataFrame
        plt.clf()
        return n_peaks, atac_kde_df

### get THRESHOLD_RNA_MAX_MITO and THRESHOLD_ATAC_MAX_MITO
### get THRESHOLD_RNA_MAX_MITO
def thresholds_on_2d_matrix(x, y, bins=150, n_classes = 4, chosen_c = 1): # Create a 2D array representation
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins) # the smaller bins is, the smoother the heatmap would be. bins=150 was chosen after testing 50, 100, 150, 200 and 300

    smooth = ski.filters.gaussian(heatmap, sigma=2) #use Gaussian filtering to smooth out the data points that do not cluster together
    thresh = smooth > threshold_multiotsu(image=smooth, classes = n_classes)[chosen_c] #use Multi-Otsu to estimate a threshold that marks foreground and background in the image `smooth`
    labels = ski.morphology.label(thresh)
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    thresh[labels != background] = 255
    heatmap_seg = thresh
        
    white_mask = heatmap_seg.T.astype(bool) & (heatmap_seg.T != 0)
    labels = measure.label(white_mask, connectivity=2)
    props = measure.regionprops(labels)
        
    largest_region = max(props, key=lambda x: x.area)
    largest_mask = labels == largest_region.label
            
    true_indices = np.argwhere(np.any(largest_mask, axis=1))

    if true_indices.size > 0:
        max_true_row_index = np.max(true_indices) # Get the highest row index that contains True
        extent_top = yedges[-1] # Convert this row index to the corresponding y-coordinate using extent
        extent_bottom = yedges[0]
        number_of_rows = heatmap_seg.shape[0]
        # Calculate the y-coordinate
        max_y_coordinate = extent_bottom + (extent_top - extent_bottom) * (max_true_row_index / (number_of_rows - 1))
        
        coords = np.column_stack(np.where(largest_mask)) # Get coordinates of all points in the largest region
        min_x = coords[:, 1].min()
        min_x_coordinate = xedges[min_x]
    else:
        max_y_coordinate = "None"
        min_x_coordinate = "None"
    
    return min_x_coordinate, max_y_coordinate
            
def get_chrMT_threshold_RNA(metrics, n_peaks): 
    '''
    lower_thres and n_peaks should be returned by guess_n_classes()
    '''
    upper_n_barcodes = len(metrics[metrics.filter_rna_min_umi == True])
    
    if n_peaks == 1 and upper_n_barcodes > 2000:
        print("RNA module chrMT thresholding: Will log transform %chrMT.")
        # Subset the nuclei to those that passed both emptydrops and post-CB nUMI thresholds, and have 0 < %chrMT < 40% to determine the %chrMT threshold. 40% is used since %chrMT per nucleus/cell should be below this threshold in practice https://pmc.ncbi.nlm.nih.gov/articles/PMC8599307/
        x = np.log10(metrics[(metrics.filter_rna_emptyDrops == True) & 
                            (metrics.filter_rna_min_umi == True) &
                            (metrics.rna_percent_mitochondrial > 1) &
                            (metrics.rna_percent_mitochondrial < 40) &
                            (metrics.filter_pct_cellbender_removed == True)].rna_umis)
        y = np.log10(metrics[(metrics.filter_rna_emptyDrops == True) & 
                    (metrics.filter_rna_min_umi == True) &
                    (metrics.rna_percent_mitochondrial > 1) &
                    (metrics.rna_percent_mitochondrial < 40) &
                    (metrics.filter_pct_cellbender_removed == True)].rna_percent_mitochondrial)
        
        min_x_coordinate, max_y_coordinate = thresholds_on_2d_matrix(x, y)
    elif n_peaks == 1 and upper_n_barcodes <= 2000:
        print("RNA module chrMT thresholding: Will *not* log transform %chrMT.")
        x = np.log10(metrics[(metrics.filter_rna_emptyDrops == True) & 
                             (metrics.filter_rna_min_umi == True) &
                             (metrics.rna_percent_mitochondrial > 0) & #if data too sparse, include more data points
                             (metrics.rna_percent_mitochondrial < 40) &
                             (metrics.filter_pct_cellbender_removed == True)].rna_umis)
        y = metrics[(metrics.filter_rna_emptyDrops == True) &
                    (metrics.filter_rna_min_umi == True) &
                    (metrics.rna_percent_mitochondrial > 0) &
                    (metrics.rna_percent_mitochondrial < 40) &
                    (metrics.filter_pct_cellbender_removed == True)].rna_percent_mitochondrial
        
        min_x_coordinate, max_y_coordinate = thresholds_on_2d_matrix(x, y)
    
    if (n_peaks == 1 and max_y_coordinate == "None") or n_peaks > 1:
        print("RNA module chrMT thresholding: Use Multi-Otsu on 1D array of chrMT.")
        THRESHOLD_RNA_MAX_MITO = estimate_threshold(metrics[(metrics.filter_rna_emptyDrops == True) &
                                                            (metrics.rna_percent_mitochondrial > 1) &
                                                            (metrics.rna_percent_mitochondrial < 40) &
                                                            (metrics.filter_pct_cellbender_removed == True)].rna_percent_mitochondrial.astype(float), classes = n_peaks+1)
    elif n_peaks == 1 and upper_n_barcodes >= 2000 and max_y_coordinate != "None":
        THRESHOLD_RNA_MAX_MITO = round(pow(10, max_y_coordinate), 2)
    elif n_peaks == 1 and upper_n_barcodes < 2000 and max_y_coordinate != "None":
        max_y_coordinate = np.log10(max_y_coordinate)
        THRESHOLD_RNA_MAX_MITO = round(pow(10, max_y_coordinate), 2)
        
    if THRESHOLD_RNA_MAX_MITO < 5:
        print("THRESHOLD_RNA_MAX_MITO guessed as < 5, set it to be 5%")
        THRESHOLD_RNA_MAX_MITO = 5 #if THRESHOLD_RNA_MAX_MITO is very low, set it to be 5%
        
    return THRESHOLD_RNA_MAX_MITO

### THRESHOLD_ATAC_MAX_MITO
def get_chrMT_threshold_ATAC(metrics, n_peaks):
    '''
    lower_thres and n_peaks should be returned by guess_n_classes()
    '''
    if n_peaks == 1:
        print("ATAC module chrMT thresholding: Will *not* log transform %chrMT.")
        # Subset the nuclei to those that passed both emptydrops and post-CB nUMI thresholds, and have 1% < %chrMT < 50% to determine the %chrMT threshold. 50% is used since %chrMT per nucleus/cell should be below this threshold in practice https://pmc.ncbi.nlm.nih.gov/articles/PMC8599307/
        x = np.log10(metrics[(metrics.filter_atac_min_hqaa == True) &
                            (metrics.atac_percent_mitochondrial > 0) &
                            (metrics.atac_percent_mitochondrial < 40)].atac_hqaa)
        y = metrics[(metrics.filter_atac_min_hqaa == True) &
                    (metrics.atac_percent_mitochondrial > 0) &
                    (metrics.atac_percent_mitochondrial < 40)].atac_percent_mitochondrial

        min_x_coordinate, max_y_coordinate = thresholds_on_2d_matrix(x, y)

    if (n_peaks == 1 and max_y_coordinate == "None") or n_peaks > 1:
        print("ATAC module chrMT thresholding: Use Multi-Otsu on 1D array of chrMT.")
        THRESHOLD_ATAC_MAX_MITO = estimate_threshold(metrics[(metrics.filter_atac_min_hqaa == True) &
                                                            (metrics.atac_percent_mitochondrial > 0) &
                                                            (metrics.atac_percent_mitochondrial < 40)].atac_percent_mitochondrial.astype(float), classes = n_peaks+1, log_scale=False)
    elif n_peaks == 1 and max_y_coordinate != "None":
        max_y_coordinate = np.log10(max_y_coordinate)
        THRESHOLD_ATAC_MAX_MITO = round(pow(10, max_y_coordinate), 2)

    if THRESHOLD_ATAC_MAX_MITO < 5:
        print("THRESHOLD_ATAC_MAX_MITO guessed as < 5, set it to be 5%")
        THRESHOLD_ATAC_MAX_MITO = 5 #if THRESHOLD_ATAC_MAX_MITO is very low, set it to be 5

    return THRESHOLD_ATAC_MAX_MITO

### functions to get CellBender-related thresholds: %ambient removed and post-CB nUMIs
def guess_n_classes_cellbender(metrics):
    data = metrics[#(metrics.cell_probability > 0.99) &
                   #(metrics.filter_rna_emptyDrops == True) &
                   #(metrics.filter_rna_min_umi == True) &
                   (metrics.pct_cellbender_removed > 5) & # assuming metrics.pct_cellbender_removed < 5 is good
                   (metrics.pct_cellbender_removed < 50) &
                   (np.isnan(metrics.pct_cellbender_removed) == False)].pct_cellbender_removed.astype(float)

    kde = sns.kdeplot(data) # Generate KDE object from the data
    # The plotted data is stored in kde.lines[0].get_xdata() and .get_ydata()
    x = kde.lines[0].get_xdata()
    y = kde.lines[0].get_ydata()

    peaks, _ = find_peaks(y, prominence=abs(max(y) * 0.05))
    n_peaks = len(peaks)

    cb_kde_df = pd.DataFrame({'x': x, 'density': y}) # Store in DataFrame for later plots
    plt.clf() # Clear the plot

    return peaks, n_peaks, cb_kde_df

def get_cellbender_thresholds(metrics, peaks_cb, n_peaks_cb, cb_kde_df): 
    '''
    n_peaks should be returned by guess_n_classes_cellbender().
    '''
    print('n_class % ambient cellbender removed {:,}'.format(n_peaks_cb))
    if n_peaks_cb == 1:
        x = np.log10(metrics[(metrics.pct_cellbender_removed > 5) &
                             (metrics.pct_cellbender_removed < 50) &
                             (np.isnan(metrics.pct_cellbender_removed) == False)].post_cellbender_umis)
        y = metrics[(metrics.pct_cellbender_removed > 5) &
                    (metrics.pct_cellbender_removed < 50) &
                    (np.isnan(metrics.pct_cellbender_removed) == False)].fraction_cellbender_removed
        min_x_coordinate, max_y_coordinate = thresholds_on_2d_matrix(x, y, chosen_c=0)
        THRESHOLD_FRACTION_CB_REMOVED = round(max_y_coordinate, 2)
        THRESHOLD_POST_CB_UMIS = round(pow(10, min_x_coordinate))

        if THRESHOLD_FRACTION_CB_REMOVED < 0.2:
            THRESHOLD_FRACTION_CB_REMOVED = 0.2
    else:
        THRESHOLD_FRACTION_CB_REMOVED = round(int(cb_kde_df.x[np.where(cb_kde_df.density == 
                                                                 cb_kde_df.density[peaks_cb[len(peaks_cb)-2]:peaks_cb[len(peaks_cb)-1]].min())[0]])/100, 2)
        THRESHOLD_POST_CB_UMIS = estimate_threshold(metrics[(metrics.pct_cellbender_removed > 5) &
                                                            (metrics.pct_cellbender_removed < 50) &
                                                            (np.isnan(metrics.pct_cellbender_removed) == False)].post_cellbender_umis.astype(float),
                                                  classes = 2)
        
    return THRESHOLD_FRACTION_CB_REMOVED, THRESHOLD_POST_CB_UMIS


## function helper to estimate RNA UMI threshold
import math
def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n / multiplier) * multiplier


### function to get exon/full gene ratio threshold:
def guess_n_classes_exon_fullgene(metrics):
    data = metrics[(metrics.rna_exon_to_full_gene_body_ratio>0.5)&
              (metrics.rna_exon_to_full_gene_body_ratio<1.0)].rna_exon_to_full_gene_body_ratio.astype(float)

    kde = sns.kdeplot(data) # Generate KDE object from the data
    # The plotted data is stored in kde.lines[0].get_xdata() and .get_ydata()
    x = kde.lines[0].get_xdata()
    y = kde.lines[0].get_ydata()

    peaks_exon, _ = find_peaks(y, prominence=abs(max(y) * 0.05))
    n_peaks_exon = len(peaks_exon)

    exon_kde_df = pd.DataFrame({'x': x, 'density': y}) # Store in DataFrame for later plots
    plt.clf() # Clear the plot

    return peaks_exon, n_peaks_exon, exon_kde_df

def get_exon_fullgene_ratio(x, y):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50) # the smaller bins is, the smoother the heatmap would be. bins=150 was chosen after testing 50, 100, 150, 200 and 300

    smooth = ski.filters.gaussian(heatmap, sigma=2) #use Gaussian filtering to smooth out the data points that do not cluster together
    thresh = smooth > threshold_multiotsu(image=smooth, classes = 4)[1] #use Multi-Otsu to estimate a threshold that marks foreground and background in the image `smooth`
    labels = ski.morphology.label(thresh)
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    thresh[labels != background] = 255
    heatmap_seg = thresh
    # mask_T is the transpose of heatmap_seg as clusters are along y axis
    mask_T = heatmap_seg.T
    y_bins_has_white = np.any(mask_T == True, axis=1)
    white_indices = np.where(y_bins_has_white)[0]

    if (len(white_indices) > 0):
        gaps = np.diff(white_indices)
        gap_bins = np.where(gaps > 1)[0]
        if (len(gap_bins) == 0):
            ends = white_indices[len(white_indices)-1]
            max_y_coordinate = (1-yedges[ends])/5+yedges[ends]
        else:
            starts = white_indices[gap_bins]
            ends = white_indices[gap_bins + 1]
            gaps_y = [(yedges[starts[i]+1], yedges[ends[i]]) for i in range(len(starts))]
            max_y_coordinate = (gaps_y[len(gaps_y)-1][1] - gaps_y[len(gaps_y)-1][0])/5+gaps_y[len(gaps_y)-1][0]
        white_indices_THRESHOLD_EXON_GENE_BODY_RATIO = max_y_coordinate
    else:
        data = metrics[(metrics.rna_exon_to_full_gene_body_ratio>0)&
                  (metrics.rna_exon_to_full_gene_body_ratio<1.0)].rna_exon_to_full_gene_body_ratio.astype(float).values
        white_indices_THRESHOLD_EXON_GENE_BODY_RATIO = threshold_multiotsu(data, classes=3)[1]

    THRESHOLD_EXON_GENE_BODY_RATIO = round(white_indices_THRESHOLD_EXON_GENE_BODY_RATIO, 2)

    return THRESHOLD_EXON_GENE_BODY_RATIO


######## functions to plot
def barcode_rank_plot(metrics, ax):
    df = metrics.sort_values('rna_umis', ascending=False)
    df['barcode_rank'] = range(1, len(df) + 1)
    sns.scatterplot(x='barcode_rank', y='rna_umis', data=df, ax=ax, hue='pass_all_filters', palette={True: 'red', False: 'black'}, edgecolor=None, alpha=0.2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Barcode rank')
    ax.set_ylabel('UMIs')
    return ax


def rna_umis_vs_rna_mito_plot(metrics, ax):
    sns.scatterplot(x='rna_umis', y='rna_fraction_mitochondrial', data=metrics, ax=ax, hue='pass_all_filters', palette={True: 'red', False: 'black'}, edgecolor=None, alpha=0.02, s=3)
    ax.set_xscale('log')
    ax.set_xlabel('UMIs')
    ax.set_ylabel('Fraction mito. (RNA)')
    return ax


def rna_umis_vs_exon_to_full_gene_body_ratio(metrics, ax):
    sns.scatterplot(x='rna_umis', y='rna_exon_to_full_gene_body_ratio', data=metrics, ax=ax, hue='pass_all_filters', palette={True: 'red', False: 'black'}, edgecolor=None, alpha=0.02, s=3)
    ax.set_xscale('log')
    ax.set_xlabel('UMIs')
    ax.set_ylabel('Exon/full-gene-body ratio (RNA)')
    return ax


def cellbender_fraction_removed(metrics, ax):
    sns.scatterplot(x='rna_umis', y='fraction_cellbender_removed', data=metrics, ax=ax, hue='pass_all_filters', palette={True: 'red', False: 'black'}, edgecolor=None, alpha=0.05)
    ax.set_xscale('log')
    ax.set_xlabel('UMIs')
    ax.set_ylabel('Fraction ambient')
    return ax


def cellbender_cell_probabilities(metrics, ax):
    sns.histplot(x='cell_probability', data=metrics[(metrics.filter_rna_emptyDrops) & (metrics.filter_rna_max_mito)], ax=ax, bins=20)
    ax.set_xlabel('Cellbender cell prob.\nfor cells by EmptyDrops and mito. thresholds')
    return ax


def rna_umis_vs_atac_hqaa_plot(metrics, ax):
    sns.scatterplot(x='rna_umis', y='atac_hqaa', data=metrics, ax=ax, hue='pass_all_filters', palette={True: 'red', False: 'black'}, edgecolor=None, alpha=0.02, s=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('UMIs (RNA)')
    ax.set_ylabel('Pass filter reads (ATAC)')
    return ax


def atac_hqaa_vs_atac_tss_enrichment_plot(metrics, ax):
    sns.scatterplot(x='atac_hqaa', y='atac_tss_enrichment', data=metrics, ax=ax, hue='pass_all_filters', palette={True: 'red', False: 'black'}, edgecolor=None, alpha=0.02, s=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Pass filter reads (ATAC)')
    ax.set_ylabel('TSS enrichment')
    return ax

# functions to plot ATAC QC
def barcode_rank_plot_atac(metrics, ax, hue='pass_all_filters', alpha=0.2):
    df = metrics.sort_values('atac_hqaa', ascending=False)
    df['barcode_rank'] = range(1, len(df) + 1)
    sns.scatterplot(x='barcode_rank', y='atac_hqaa', data=df, ax=ax, hue=hue, palette={True: 'red', False: 'black'}, edgecolor=None, alpha=alpha)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Barcode rank')
    ax.set_ylabel('Pass filter reads (ATAC)')
    return ax

def atac_hqaa_vs_atac_tss_enrichment_plot(metrics, ax, hue='pass_all_filters', alpha=0.2):
    sns.scatterplot(x='atac_hqaa', y='atac_tss_enrichment', data=metrics, ax=ax, hue=hue, palette={True: 'red', False: 'black'}, edgecolor=None, alpha=alpha, s=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Pass filter reads (ATAC)')
    ax.set_ylabel('TSS enrichment')
    return ax

def atac_hqaa_vs_atac_mt_pct_plot(metrics, ax, hue='pass_all_filters', alpha=0.2):
    sns.scatterplot(x='atac_hqaa', y='atac_percent_mitochondrial', data=metrics, ax=ax, hue=hue, palette={True: 'red', False: 'black'}, edgecolor=None, alpha=alpha, s=3)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlabel('Pass filter reads (ATAC)')
    ax.set_ylabel('atac_percent_mitochondrial')
    return ax

def atac_tss_enrichment_vs_atac_mt_pct_plot(metrics, ax, hue='pass_all_filters', alpha=0.2):
    sns.scatterplot(x='atac_tss_enrichment', y='atac_percent_mitochondrial', data=metrics, ax=ax, hue=hue, palette={True: 'red', False: 'black'}, edgecolor=None, alpha=alpha, s=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('tss_enrichment')
    ax.set_ylabel('Fraction mito. (ATAC)')
    return ax


