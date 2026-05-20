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
import logging
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter


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


##### LOGGER
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "\n" + "=" * 60 + "\n%(levelname)s: %(message)s\n" + "=" * 60 + "\n"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

##### FUNCTIONS FOR MITO THRESHOLDS
from scipy.signal import find_peaks
import skimage as ski
from scipy import ndimage as ndi
from skimage import measure
import matplotlib.pyplot as plt

def guess_n_classes(metrics, mode = 'RNA'):
    """
    Guess the number of classes (peaks) in the mitochondrial percentage distribution.

    Parameters
    ----------
    metrics : pd.DataFrame
        DataFrame containing QC metrics.
    mode : str
        Either 'RNA' or 'ATAC'.

    Returns
    -------
    tuple
        (n_peaks, kde_df) where n_peaks is the number of prominent peaks
        and kde_df is a DataFrame with the KDE x and density values.
    """
    modes = ['RNA', 'ATAC']
    if mode not in modes:
        raise ValueError("Invalid mode. Expected one of: %s" % modes)

    if mode == "RNA":
        required_columns = {
            "filter_rna_emptyDrops",
            "filter_rna_min_umi",
            "rna_percent_mitochondrial",
            "filter_pct_cellbender_removed",
        }
        pct_mito_col = "rna_percent_mitochondrial"
    else:
        required_columns = {"filter_atac_min_hqaa", "atac_percent_mitochondrial"}
        pct_mito_col = "atac_percent_mitochondrial"
    
    # Validate required columns exist
    missing_columns = required_columns - set(metrics.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in metrics DataFrame: {missing_columns}"
        )
    
    ## Step 1: guess lower bound to preclude
    if mode == "RNA":
        data = metrics[(metrics.filter_rna_emptyDrops == True) &
                       (metrics.filter_rna_min_umi == True) &
                       (metrics.rna_percent_mitochondrial > 1) &
                       (metrics.rna_percent_mitochondrial < 50) & # assuming cells/nuclei with > 50% mitochondrial reads has too high %chrMT
                       (metrics.filter_pct_cellbender_removed == True)].rna_percent_mitochondrial.astype(float)
    else:
        data = metrics[(metrics.filter_atac_min_hqaa == True) &
                       (metrics.atac_percent_mitochondrial > 1) &
                       (metrics.atac_percent_mitochondrial < 50)].atac_percent_mitochondrial.astype(float)

    values = np.log10(data).values
    values = values.reshape((len(values),1))
    thresholds = threshold_multiotsu(image=values, classes=4, nbins=256)
    thresholds = [pow(10, i) for i in thresholds] # convert back to linear scale

    # Step 2: get density and check number of distributions
    if mode == "RNA":
        data = np.log10(metrics[(metrics.filter_rna_emptyDrops == True) &
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
    logger.info(
        f"Number of prominent peaks in {mode} %chrMT distribution: {n_peaks:,}"
    )

    if mode == "RNA":
        rna_kde_df = pd.DataFrame({'x': x, 'density': y}) # Store in DataFrame for later plots
        plt.clf() # Clear the plot
        return n_peaks, rna_kde_df
    else:
        atac_kde_df = pd.DataFrame({'x': x, 'density': y}) # Store in DataFrame
        plt.clf()
        return n_peaks, atac_kde_df

### get THRESHOLD_RNA_MAX_MITO and THRESHOLD_ATAC_MAX_MITO
### get THRESHOLD_RNA_MAX_MITO
def thresholds_on_2d_matrix(x, y, bins=150, n_classes = 4, chosen_class = 1): # Create a 2D array representation
    """
    Estimate thresholds from a 2D histogram using Multi-Otsu segmentation.

    Creates a 2D histogram heatmap, applies Gaussian smoothing and Multi-Otsu
    thresholding to segment foreground from background, then identifies the
    largest connected region and returns its boundary coordinates.

    Parameters
    ----------
    x : array-like
        Values for the x-axis of the 2D histogram.
    y : array-like
        Values for the y-axis of the 2D histogram.
    bins : int, optional
        Number of bins for the 2D histogram. Smaller values produce smoother
        heatmaps. Default is 150 (chosen after testing 50, 100, 150, 200, 300).
    n_classes : int, optional
        Number of classes for Multi-Otsu thresholding. Default is 4.
    chosen_class : int, optional
        Index of the Multi-Otsu threshold to use for binarization. Default is 1.

    Returns
    -------
    tuple
        (min_x_coordinate, max_y_coordinate) of the largest foreground region,
        or (None, None) if no foreground region is found.
    """

    # Step 1: Build 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins) # the smaller bins is, the smoother the heatmap would be. bins=150 was chosen after testing 50, 100, 150, 200 and 300
    
    # Step 2: Smooth the heatmap with a Gaussian filter
    smooth = ski.filters.gaussian(heatmap, sigma=2) #use Gaussian filtering to smooth out the data points that do not cluster together

    # Step 2: Smooth the heatmap with a Gaussian filter
    smoothed = ski.filters.gaussian(heatmap, sigma=2)

    # Step 3: Apply Multi-Otsu thresholding to separate foreground/background
    otsu_thresholds = threshold_multiotsu(image=smoothed, classes=n_classes)
    binary_mask = smoothed > otsu_thresholds[chosen_class]

    # Step 4: Label connected components and remove non-background regions
    labels = ski.morphology.label(binary_mask)
    label_counts = np.bincount(labels.ravel())
    background_label = np.argmax(label_counts)
    binary_mask[labels != background_label] = True

    # Step 5: Find the largest connected foreground region (transposed view)
    foreground_mask = binary_mask.T.astype(bool) & (binary_mask.T != 0)
    region_labels = measure.label(foreground_mask, connectivity=2)
    region_props = measure.regionprops(region_labels)

    if not region_props:
        return None, None

    largest_region = max(region_props, key=lambda r: r.area)
    largest_mask = region_labels == largest_region.label

    # Step 6: Extract boundary coordinates from the largest region
    true_row_indices = np.argwhere(np.any(largest_mask, axis=1))

    if true_row_indices.size == 0:
        return None, None

    # Convert the highest row index to a y-coordinate
    max_row_index = np.max(true_row_indices)
    n_rows = binary_mask.shape[0]
    max_y_coordinate = yedges[0] + (yedges[-1] - yedges[0]) * (
        max_row_index / (n_rows - 1)
    )

    # Convert the minimum column index to an x-coordinate
    region_coords = np.column_stack(np.where(largest_mask))
    min_col_index = region_coords[:, 1].min()
    min_x_coordinate = xedges[min_col_index]

    return min_x_coordinate, max_y_coordinate
            
def get_chrMT_threshold_RNA(metrics, n_peaks): 
    """
    Determine the RNA mitochondrial percentage threshold.

    Uses a 2D histogram segmentation approach when a single peak is detected,
    falling back to 1D Multi-Otsu thresholding when multiple peaks are present
    or when the 2D method fails.

    Parameters
    ----------
    metrics : pd.DataFrame
        QC metrics DataFrame. Expected columns: filter_rna_emptyDrops,
        filter_rna_min_umi, rna_percent_mitochondrial,
        filter_pct_cellbender_removed, rna_umis.
    n_peaks : int
        Number of prominent peaks returned by `guess_n_classes()`.

    Returns
    -------
    float
        The estimated maximum mitochondrial percentage threshold for RNA.
    """
    upper_n_barcodes = len(metrics[metrics.filter_rna_min_umi == True])
    
    if n_peaks == 1 and upper_n_barcodes > 2000:
        logger.info("RNA chrMT thresholding: Will log-transform %%chrMT.")
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
        logger.info("RNA chrMT thresholding: Will *not* log-transform %%chrMT.")
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
        logger.info("RNA chrMT thresholding: Using Multi-Otsu on 1D array of chrMT.")
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
        logger.info("THRESHOLD_RNA_MAX_MITO guessed as < 5, set it to be 5%%.")
        THRESHOLD_RNA_MAX_MITO = 5 #if THRESHOLD_RNA_MAX_MITO is very low, set it to be 5%
        
    return THRESHOLD_RNA_MAX_MITO

### THRESHOLD_ATAC_MAX_MITO
def get_chrMT_threshold_ATAC(metrics, n_peaks):
    """
    Determine the ATAC mitochondrial percentage threshold.

    Uses a 2D histogram segmentation approach when a single peak is detected,
    falling back to 1D Multi-Otsu thresholding when multiple peaks are present
    or when the 2D method fails.

    Parameters
    ----------
    metrics : pd.DataFrame
        QC metrics DataFrame. Expected columns: filter_atac_min_hqaa, 
        atac_percent_mitochondrial, atac_hqaa, atac_percent_mitochondrial.
    n_peaks : int
        Number of prominent peaks returned by `guess_n_classes()`.

    Returns
    -------
    float
        The estimated maximum mitochondrial percentage threshold for ATAC.
    """
    if n_peaks == 1:
        logger.info("ATAC module chrMT thresholding: Will *not* log transform %%chrMT.")
        # Subset the nuclei to those that passed both emptydrops and post-CB nUMI thresholds, and have 1% < %chrMT < 40% to determine the %chrMT threshold. 40% is used since %chrMT per nucleus/cell should be below this threshold in practice https://pmc.ncbi.nlm.nih.gov/articles/PMC8599307/
        x = np.log10(metrics[(metrics.filter_atac_min_hqaa == True) &
                            (metrics.atac_percent_mitochondrial > 0) &
                            (metrics.atac_percent_mitochondrial < 40)].atac_hqaa)
        y = metrics[(metrics.filter_atac_min_hqaa == True) &
                    (metrics.atac_percent_mitochondrial > 0) &
                    (metrics.atac_percent_mitochondrial < 40)].atac_percent_mitochondrial

        min_x_coordinate, max_y_coordinate = thresholds_on_2d_matrix(x, y)

    if (n_peaks == 1 and max_y_coordinate == "None") or n_peaks > 1:
        logger.info("ATAC module chrMT thresholding: Use Multi-Otsu on 1D array of chrMT.")
        THRESHOLD_ATAC_MAX_MITO = estimate_threshold(metrics[(metrics.filter_atac_min_hqaa == True) &
                                                            (metrics.atac_percent_mitochondrial > 0) &
                                                            (metrics.atac_percent_mitochondrial < 40)].atac_percent_mitochondrial.astype(float), classes = n_peaks+1, log_scale=False)
    elif n_peaks == 1 and max_y_coordinate != "None":
        max_y_coordinate = np.log10(max_y_coordinate)
        THRESHOLD_ATAC_MAX_MITO = round(pow(10, max_y_coordinate), 2)

    if THRESHOLD_ATAC_MAX_MITO < 10:
        logger.info("THRESHOLD_ATAC_MAX_MITO guessed as < 10, set it to be 10%%.")
        THRESHOLD_ATAC_MAX_MITO = 10 #if THRESHOLD_ATAC_MAX_MITO is very low, set it to be 10

    return THRESHOLD_ATAC_MAX_MITO

### functions to get CellBender-related thresholds: %ambient removed and post-CB nUMIs
def guess_n_classes_cellbender(metrics):
    """
    Estimate the number of distribution classes in CellBender removal percentages.

    Filters the data to a plausible range, fits a kernel density estimate,
    and counts prominent peaks to infer the number of underlying distributions.

    Parameters
    ----------
    metrics : pd.DataFrame
        QC metrics DataFrame. Must contain a 'pct_cellbender_removed' column.

    Returns
    -------
    tuple
        (peak_indices, n_peaks, kde_df) where:
        - peak_indices : np.ndarray of indices into the KDE x-array where peaks occur
        - n_peaks : int, number of prominent peaks detected
        - kde_df : pd.DataFrame with columns 'x' and 'density' for downstream plotting
    """

    data = metrics[(metrics.pct_cellbender_removed > 5) & # assuming metrics.pct_cellbender_removed < 5 is good
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
    """
    Determine CellBender removal thresholds based on peak structure.

    Uses a 2D histogram segmentation approach when a single peak is detected,
    or identifies the density minimum between the last two peaks for multi-peak
    distributions.

    Parameters
    ----------
    metrics : pd.DataFrame
        QC metrics DataFrame. Expected columns: pct_cellbender_removed,
        post_cellbender_umis, fraction_cellbender_removed.
    peak_indices : np.ndarray
        Indices of detected peaks in the KDE density curve, returned by
        `guess_n_classes_cellbender()`.
    n_peaks : int
        Number of prominent peaks detected, returned by
        `guess_n_classes_cellbender()`.
    kde_df : pd.DataFrame
        DataFrame with 'x' and 'density' columns from the KDE fit,
        returned by `guess_n_classes_cellbender()`.

    Returns
    -------
    tuple
        (threshold_fraction_cb_removed, threshold_post_cb_umis) where:
        - threshold_fraction_cb_removed : float, maximum allowed fraction of
          counts removed by CellBender
        - threshold_post_cb_umis : float, minimum UMI count after CellBender
    """
    logger.info(f"Number of classes in %% ambient CellBender removed: {n_peaks_cb:,}")
    if n_peaks_cb == 1:
        x = np.log10(metrics[(metrics.pct_cellbender_removed > 5) &
                             (metrics.pct_cellbender_removed < 50) &
                             (np.isnan(metrics.pct_cellbender_removed) == False)].post_cellbender_umis)
        y = metrics[(metrics.pct_cellbender_removed > 5) &
                    (metrics.pct_cellbender_removed < 50) &
                    (np.isnan(metrics.pct_cellbender_removed) == False)].fraction_cellbender_removed
        min_x_coordinate, max_y_coordinate = thresholds_on_2d_matrix(x, y, chosen_class=0)
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
    """
    Estimate the number of distribution classes in exon-to-full-gene-body ratio.

    Filters the data to a plausible range, fits a kernel density estimate,
    and counts prominent peaks to infer the number of underlying distributions.

    Parameters
    ----------
    metrics : pd.DataFrame
        QC metrics DataFrame. Must contain 'rna_exon_to_full_gene_body_ratio'.

    Returns
    -------
    tuple
        (peak_indices, n_peaks, kde_df) where:
        - peaks_exon : np.ndarray of indices into the KDE x-array where peaks occur
        - n_peaks_exon : int, number of prominent peaks detected
        - exon_kde_df : pd.DataFrame with columns 'x' and 'density' for downstream plotting
    """
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
    """
    Estimate the exon-to-full-gene-body ratio threshold using 2D segmentation.

    Creates a 2D histogram of the input data, applies Gaussian smoothing and
    Multi-Otsu thresholding, then identifies gaps between foreground clusters
    along the y-axis to determine a threshold. Falls back to 1D Multi-Otsu
    if no foreground regions are detected.

    Parameters
    ----------
    x : array-like
        Values for the x-axis (e.g., log10 UMI counts).
    y : array-like
        Values for the y-axis (e.g., exon-to-full-gene-body ratio).
    metrics : pd.DataFrame, optional
        Full metrics DataFrame, required for the 1D fallback method.
        Must contain 'rna_exon_to_full_gene_body_ratio'.

    Returns
    -------
    float
        Estimated exon-to-full-gene-body ratio threshold, rounded to 2 decimals.
    """
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

######## functions for knee plot analysis
# Savitzky-Golay filter parameters
_MIN_WINDOW_LENGTH = 201
_WINDOW_DIVISOR = 5
_SAVGOL_POLYORDER = 1

# Peak detection parameters
_PEAK_PROMINENCE_FRACTION = 0.1

# Minimum UMI threshold for peak detection
_MIN_UMIS_FOR_PEAKS = 5


def classify_umi_range(umis, end_cliff, knee):
    """
    Classify a UMI count into a labeled range for plotting.

    Parameters
    ----------
    umis : float
        UMI count for a barcode.
    end_cliff : float
        UMI value marking the end of the cliff region.
    knee : float
        UMI value marking the knee point.

    Returns
    -------
    str
        Label indicating which range the UMI count falls into.
    """
    if umis < end_cliff:
        return f"UMIs < {end_cliff}"
    elif umis < knee:
        return f"{end_cliff} < UMIs < {knee}"
    else:
        return f"UMIs > {knee}"


def analyze_knee_plot(
    metrics, knee, knee_rank, end_cliff, end_cliff_rank, inflection_rank
):
    """
    Analyze the barcode rank (knee) plot to detect transition points.

    Performs interpolation and Savitzky-Golay smoothing on the rank-UMI curve
    to identify significant slope changes, then detects peaks in the smoothed
    derivative to assess knee plot quality.

    Parameters
    ----------
    metrics : pd.DataFrame
        QC metrics DataFrame. Must contain 'rna_umis'.
    knee : float
        UMI value at the knee point.
    knee_rank : int
        Barcode rank corresponding to the knee point.
    end_cliff : float
        UMI value at the end of the cliff region.
    end_cliff_rank : int
        Barcode rank corresponding to the end of the cliff.
    inflection_rank : int
        Barcode rank corresponding to the inflection point.

    Returns
    -------
    tuple
        (df_ranked, df_interpolated, n_peaks, final_peak_indices) where:
        - df_ranked : pd.DataFrame with ranked barcodes and UMI changes
        - df_interpolated : pd.DataFrame with interpolated and smoothed values
        - n_peaks : int, number of detected knee points
        - final_peak_indices : np.ndarray, indices of the final peaks
    """
    # Step 1: Rank barcodes by UMI count
    df_ranked = _rank_barcodes(metrics, end_cliff, knee)

    # Step 2: Interpolate to log-uniform spacing
    df_interpolated = _interpolate_log_uniform(df_ranked, end_cliff, knee)

    # Step 3: Smooth with Savitzky-Golay filter
    window_size = _compute_window_size(
        df_interpolated, knee_rank, inflection_rank
    )
    df_interpolated = _apply_savgol_smoothing(df_interpolated, window_size)

    # Step 4: Detect peaks in the smoothed derivative
    n_peaks, final_peak_indices = _detect_knee_peaks(
        df_interpolated, window_size, end_cliff_rank
    )

    return df_ranked, df_interpolated, n_peaks, final_peak_indices


def _rank_barcodes(metrics, end_cliff, knee):
    """
    Sort barcodes by UMI count and compute discrete differences.

    Parameters
    ----------
    metrics : pd.DataFrame
    end_cliff : float
    knee : float

    Returns
    -------
    pd.DataFrame
        Ranked DataFrame with 'barcode_rank', 'range', and 'change_umis' columns.
    """
    df = (
        metrics.sort_values("rna_umis", ascending=False)
        .assign(barcode_rank=lambda d: range(1, len(d) + 1))
        .query("rna_umis > 0")  # Avoid log10(0)
        .copy()
    )

    df["range"] = df["rna_umis"].apply(
        classify_umi_range, end_cliff=end_cliff, knee=knee
    )

    # Compute discrete differences in log-space
    log_umis = np.log10(df["rna_umis"].values)
    change = np.abs(np.diff(log_umis, prepend=log_umis[0]))
    df["change_umis"] = change

    return df


def _interpolate_log_uniform(df_ranked, end_cliff, knee):
    """
    Interpolate the rank-UMI curve onto log-uniformly spaced points.

    Equal spacing in log-space is required for meaningful Savitzky-Golay
    filtering downstream.

    Parameters
    ----------
    df_ranked : pd.DataFrame
        Must contain 'barcode_rank' and 'rna_umis'.
    end_cliff : float
    knee : float

    Returns
    -------
    pd.DataFrame
        Interpolated DataFrame with 'x_new', 'y_new', and 'range' columns.
    """
    log_rank = np.log10(df_ranked["barcode_rank"].values)
    log_umis = np.log10(df_ranked["rna_umis"].values)

    interpolator = interp1d(x=log_rank, y=log_umis)

    n_points = len(df_ranked) * 2
    log_rank_uniform = np.linspace(log_rank.min(), log_rank.max(), num=n_points)
    log_umis_interpolated = interpolator(log_rank_uniform)

    df_interp = pd.DataFrame({
        "x_new": 10**log_rank_uniform,
        "y_new": 10**log_umis_interpolated,
    })

    df_interp["range"] = df_interp["y_new"].apply(
        classify_umi_range, end_cliff=end_cliff, knee=knee
    )

    return df_interp


def _compute_window_size(df_interpolated, knee_rank, inflection_rank):
    """
    Compute the Savitzky-Golay window size from the knee-to-inflection range.

    The window is based on the number of interpolated points between the knee
    and inflection ranks, divided by a factor and forced to be odd.

    Parameters
    ----------
    df_interpolated : pd.DataFrame
        Must contain 'x_new'.
    knee_rank : int
    inflection_rank : int

    Returns
    -------
    int
        Odd-valued window length for the Savitzky-Golay filter.
    """
    # Ensure lower_rank <= upper_rank regardless of input order
    # Note: inflection_rank can sometimes be less than knee_rank due to
    # instability in EmptyDrops inflection estimation (one rank can map to
    # multiple inflection points while UMI values remain consistent)
    lower_rank = min(knee_rank, inflection_rank)
    upper_rank = max(knee_rank, inflection_rank)

    n_points_in_range = (
        (df_interpolated["x_new"] >= lower_rank)
        & (df_interpolated["x_new"] <= upper_rank)
    ).sum()

    window = round(n_points_in_range / _WINDOW_DIVISOR)

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    # Enforce minimum window size to filter sufficient noise
    window = max(window, _MIN_WINDOW_LENGTH)

    logger.info(f"Savitzky-Golay window size: {window}")

    return window


def _apply_savgol_smoothing(df_interpolated, window_size):
    """
    Apply Savitzky-Golay smoothing and compute the derivative of the curve.

    Parameters
    ----------
    df_interpolated : pd.DataFrame
        Must contain 'y_new'.
    window_size : int
        Odd-valued window length for the filter.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added 'change' column (smoothed derivative).
    """
    log_y = np.log10(df_interpolated["y_new"].values)
    smoothed = savgol_filter(log_y, window_length=window_size, polyorder=_SAVGOL_POLYORDER)

    # Derivative of smoothed curve (preserving sign for directionality)
    derivative = np.diff(smoothed, prepend=smoothed[0])
    df_interpolated = df_interpolated.copy()
    df_interpolated["change"] = derivative

    return df_interpolated


def _detect_knee_peaks(df_interpolated, window_size, end_cliff_rank):
    """
    Detect peaks in the smoothed derivative to identify knee points.

    Applies a second round of smoothing to the derivative, then uses peak
    detection. If multiple peaks are found beyond the cliff region, only
    the most prominent one is retained.

    Parameters
    ----------
    df_interpolated : pd.DataFrame
        Must contain 'x_new', 'y_new', and 'change'.
    window_size : int
        Base window size from the first smoothing pass.
    end_cliff_rank : int
        Barcode rank at the end of the cliff region.

    Returns
    -------
    tuple
        (n_peaks, final_peak_indices) where:
        - n_peaks : int, number of detected knee points
        - final_peak_indices : np.ndarray, indices of retained peaks
    """
    # Second smoothing pass on the derivative for peak detection
    peak_window = round(window_size * 2)
    if peak_window % 2 == 0:
        peak_window += 1

    smoothed_change = savgol_filter(
        df_interpolated["change"].values,
        window_length=peak_window,
        polyorder=_SAVGOL_POLYORDER,
    )
    df_interpolated = df_interpolated.copy()
    df_interpolated["change_hat"] = smoothed_change

    # Find peaks in the absolute derivative (only where UMIs > minimum)
    above_min = df_interpolated["y_new"] > _MIN_UMIS_FOR_PEAKS
    x_values = np.abs(df_interpolated.loc[above_min, "change_hat"].values)
    min_prominence = np.abs(smoothed_change.min()) * _PEAK_PROMINENCE_FRACTION

    peak_indices, _ = find_peaks(x_values, prominence=min_prominence)
    # Map back to df_interpolated indices
    valid_indices = df_interpolated.index[above_min]
    peak_indices_global = valid_indices[peak_indices]

    # Consolidate multiple peaks beyond the cliff to keep only the strongest
    final_peak_indices = _consolidate_peaks(
        df_interpolated, peak_indices_global, end_cliff_rank
    )
    n_peaks = len(final_peak_indices)

    # Log warnings based on peak structure
    _log_knee_warnings(df_interpolated, final_peak_indices, end_cliff_rank, n_peaks)

    return n_peaks, final_peak_indices


def _consolidate_peaks(df_interpolated, peak_indices, end_cliff_rank):
    """
    Consolidate multiple post-cliff peaks into a single strongest peak.

    Parameters
    ----------
    df_interpolated : pd.DataFrame
    peak_indices : np.ndarray
        Global indices of detected peaks.
    end_cliff_rank : int

    Returns
    -------
    np.ndarray
        Consolidated array of peak indices.
    """
    if len(peak_indices) <= 1:
        return peak_indices

    peak_ranks = df_interpolated.loc[peak_indices, "x_new"]
    post_cliff_mask = peak_ranks > end_cliff_rank

    if post_cliff_mask.sum() <= 1:
        return peak_indices

    # Keep all pre-cliff peaks, but only the strongest post-cliff peak
    pre_cliff_peaks = peak_indices[~post_cliff_mask]
    post_cliff_peaks = peak_indices[post_cliff_mask]

    strongest_post_cliff = df_interpolated.loc[
        post_cliff_peaks, "change_hat"
    ].idxmin()

    return np.append(pre_cliff_peaks, strongest_post_cliff)


def _log_knee_warnings(df_interpolated, final_peak_indices, end_cliff_rank, n_peaks):
    """
    Log warnings about knee plot quality based on detected peaks.

    Parameters
    ----------
    df_interpolated : pd.DataFrame
    final_peak_indices : np.ndarray
    end_cliff_rank : int
    n_peaks : int
    """
    if n_peaks == 0:
        logger.error("Could not detect peaks of change. Check input data.")
    elif n_peaks == 1:
        peak_ranks = df_interpolated.loc[final_peak_indices, "x_new"]
        if (peak_ranks > end_cliff_rank).all():
            logger.warning("Knee plot does not show a clear knee point.")
    elif n_peaks > 2:
        logger.warning("Knee plot may have multiple knees.")

    logger.info(f"Number of prominent transitions in knee plot: {n_peaks:,}")

######## all functions for ATAC max_pct_reads_from_single_autosome
# 2D histogram parameters
_HIST_BINS = 150
_GAUSSIAN_SIGMA = 2
_OTSU_CLASSES = 4
_OTSU_THRESHOLD_INDEX = 0

# Peak detection
_PEAK_PROMINENCE_FRACTION = 0.05

# Fraction-to-percent conversion
_FRACTION_TO_PCT = 100


def get_atac_max_autosome_threshold(metrics):
    """
    Estimate the maximum fraction of reads from a single autosome threshold.

    Uses density-based peak detection to determine the number of underlying
    distributions. For single-peak distributions, applies 2D histogram
    segmentation. For multi-peak distributions, falls back to 1D Multi-Otsu.

    Parameters
    ----------
    metrics : pd.DataFrame
        QC metrics DataFrame. Must contain 'atac_max_fraction_reads_from_single_autosome',
        'filter_atac_min_hqaa', and 'hqaa' (for the 2D method).

    Returns
    -------
    tuple
        (threshold, n_peaks, kde_df) where:
        - threshold : float, estimated maximum percent reads from single autosome
        - n_peaks : int, number of peaks detected in the distribution
        - kde_df : pd.DataFrame with 'x' and 'density' columns for plotting
    """
    # Convert fraction to percentage
    metrics = metrics.copy()
    metrics["atac_max_pct_reads_from_single_autosome"] = (
        metrics["atac_max_fraction_reads_from_single_autosome"] * _FRACTION_TO_PCT
    )

    # Step 1: Detect number of peaks in the distribution
    n_peaks, kde_df = _guess_n_peaks(metrics)

    # Step 2: Estimate threshold based on peak structure
    if n_peaks == 1:
        threshold = _threshold_single_peak(metrics, n_peaks)
    else:
        threshold = _threshold_multi_peak(metrics, n_peaks)
    
    if threshold < 20:
        threshold = 20
        logger.info("thres_max_fraction_reads_from_single_autosome guessed as < 20, set it to be 20%%.")

    return threshold, n_peaks, kde_df


def _guess_n_peaks(metrics): # this function is reused a lot, worth merging -- to do
    """
    Detect the number of prominent peaks in the autosome read fraction distribution.

    Parameters
    ----------
    metrics : pd.DataFrame

    Returns
    -------
    tuple
        (n_peaks, kde_df)
    """
    filtered = metrics.loc[
        metrics["filter_atac_min_hqaa"].eq(True),
        "atac_max_pct_reads_from_single_autosome",
    ].astype(float)

    log_data = np.log10(filtered)

    # Generate KDE and extract curve
    kde_ax = sns.kdeplot(log_data)
    x = kde_ax.lines[0].get_xdata()
    y = kde_ax.lines[0].get_ydata()
    plt.clf()

    # Detect peaks
    min_prominence = np.abs(y.max()) * _PEAK_PROMINENCE_FRACTION
    peaks, _ = find_peaks(y, prominence=min_prominence)
    n_peaks = len(peaks)

    logger.info(
        f"Number of prominent peaks in ATAC max_pct_reads_from_single_autosome: {n_peaks:,}"
    )

    kde_df = pd.DataFrame({"x": x, "density": y})

    return n_peaks, kde_df


def _threshold_single_peak(metrics, n_peaks):
    """
    Estimate threshold using 2D histogram segmentation for single-peak data.

    Attempts log-transformed y-axis first. If that fails, tries linear y-axis.
    Falls back to 1D Multi-Otsu if both fail.

    Parameters
    ----------
    metrics : pd.DataFrame
    n_peaks : int

    Returns
    -------
    float
        Estimated threshold (in percent).
    """
    filtered = metrics.loc[metrics["filter_atac_min_hqaa"].eq(True)]
    x = np.log10(filtered["atac_hqaa"])

    # Attempt 1: Log-transformed y-axis
    y_log = np.log10(filtered["atac_max_pct_reads_from_single_autosome"])
    max_y = _segment_2d_and_find_max_y(x, y_log)

    if max_y is not None:
        return round(10**max_y)

    # Attempt 2: Linear y-axis
    logger.info(
        "2D segmentation failed with log-transform; retrying without log-transform."
    )
    y_linear = filtered["atac_max_pct_reads_from_single_autosome"]
    max_y = _segment_2d_and_find_max_y(x, y_linear)

    if max_y is not None:
        return round(10 ** np.log10(max_y))

    # Attempt 3: Fallback to 1D Multi-Otsu
    logger.info(
        "2D segmentation failed; falling back to 1D Multi-Otsu."
    )
    return _threshold_multi_peak(metrics, n_peaks)


def _threshold_multi_peak(metrics, n_peaks):
    """
    Estimate threshold using 1D Multi-Otsu for multi-peak distributions.

    Parameters
    ----------
    metrics : pd.DataFrame
    n_peaks : int

    Returns
    -------
    float
        Estimated threshold (in percent).
    """
    filtered_data = metrics.loc[
        metrics["filter_atac_min_hqaa"].eq(True),
        "atac_max_pct_reads_from_single_autosome",
    ].astype(float)

    return estimate_threshold(filtered_data, classes=n_peaks + 1)


def _segment_2d_and_find_max_y(x, y):
    """
    Perform 2D histogram segmentation and find the maximum y-coordinate
    of the foreground region.

    Parameters
    ----------
    x : pd.Series or np.ndarray
        X-axis values (e.g., log10 HQAA reads).
    y : pd.Series or np.ndarray
        Y-axis values (e.g., log10 or linear max_pct_reads_from_single_autosome).

    Returns
    -------
    float or None
        Maximum y-coordinate of the foreground, or None if no foreground found.
    """
    # Build 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=_HIST_BINS)

    # Smooth and threshold
    smoothed = ski.filters.gaussian(heatmap, sigma=_GAUSSIAN_SIGMA)
    otsu_thresholds = threshold_multiotsu(
        image=smoothed, classes=_OTSU_CLASSES
    )
    binary_mask = smoothed > otsu_thresholds[_OTSU_THRESHOLD_INDEX]

    # Remove non-background regions
    labels = ski.morphology.label(binary_mask)
    label_counts = np.bincount(labels.ravel())
    background_label = np.argmax(label_counts)
    binary_mask[labels != background_label] = True

    # Find the highest y-bin containing foreground (transposed view)
    foreground_rows = np.argwhere(np.any(binary_mask.T, axis=1))

    if foreground_rows.size == 0:
        return None

    # Convert row index to y-coordinate
    max_row_index = np.max(foreground_rows)
    n_rows = binary_mask.shape[0]
    max_y_coordinate = (
        yedges[0]
        + (yedges[-1] - yedges[0]) * (max_row_index / (n_rows - 1))
    )

    return max_y_coordinate
    
######## functions to plot
def barcode_rank_plot(metrics, ax):
    """
    Create a barcode rank plot (knee plot) colored by filter status.

    Barcodes are ranked by descending UMI count and plotted on log-log axes,
    with color indicating whether each barcode passed all QC filters.

    Parameters
    ----------
    metrics : pd.DataFrame
        QC metrics DataFrame. Must contain 'rna_umis' and 'pass_all_filters'.
    ax : matplotlib.axes.Axes
        Axes object on which to draw the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the plot drawn.
    """
    df = metrics.sort_values('rna_umis', ascending=False)
    df['barcode_rank'] = range(1, len(df) + 1)
    sns.scatterplot(x='barcode_rank', y='rna_umis', data=df, ax=ax, hue='pass_all_filters', palette={True: 'red', False: 'black'}, edgecolor=None, alpha=0.2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Barcode rank')
    ax.set_ylabel('UMIs')
    return ax


def rna_umis_vs_rna_mito_plot(metrics, ax):
    """
    Scatter plot of RNA UMIs vs. mitochondrial fraction.

    Parameters
    ----------
    metrics : pd.DataFrame
        Must contain 'rna_umis', 'rna_fraction_mitochondrial', 'pass_all_filters'.
    ax : matplotlib.axes.Axes
    alpha : float, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    sns.scatterplot(x='rna_umis', y='rna_fraction_mitochondrial', data=metrics, ax=ax, hue='pass_all_filters', palette={True: 'red', False: 'black'}, edgecolor=None, alpha=0.02, s=3)
    ax.set_xscale('log')
    ax.set_xlabel('UMIs')
    ax.set_ylabel('Fraction mito. (RNA)')
    return ax


def rna_umis_vs_exon_to_full_gene_body_ratio(metrics, ax):
    """
    Scatter plot of RNA UMIs vs. exon-to-full-gene-body ratio.

    Parameters
    ----------
    metrics : pd.DataFrame
        Must contain 'rna_umis', 'rna_exon_to_full_gene_body_ratio', 'pass_all_filters'.
    ax : matplotlib.axes.Axes
    alpha : float, optional

    Returns
    -------
    matplotlib.axes.Axes
    """

    sns.scatterplot(x='rna_umis', y='rna_exon_to_full_gene_body_ratio', data=metrics, ax=ax, hue='pass_all_filters', palette={True: 'red', False: 'black'}, edgecolor=None, alpha=0.02, s=3)
    ax.set_xscale('log')
    ax.set_xlabel('UMIs')
    ax.set_ylabel('Exon/full-gene-body ratio (RNA)')
    return ax


def cellbender_fraction_removed(metrics, ax):
    """
    Scatter plot of RNA UMIs vs. fraction of counts removed by CellBender.

    Parameters
    ----------
    metrics : pd.DataFrame
        Must contain 'rna_umis', 'fraction_cellbender_removed', 'pass_all_filters'.
    ax : matplotlib.axes.Axes
    alpha : float, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    sns.scatterplot(x='rna_umis', y='fraction_cellbender_removed', data=metrics, ax=ax, hue='pass_all_filters', palette={True: 'red', False: 'black'}, edgecolor=None, alpha=0.05)
    ax.set_xscale('log')
    ax.set_xlabel('UMIs')
    ax.set_ylabel('Fraction ambient')
    return ax


def cellbender_cell_probabilities(metrics, ax):
    """
    Histogram of CellBender cell probabilities for filtered barcodes.

    Only includes barcodes passing EmptyDrops and mitochondrial filters.

    Parameters
    ----------
    metrics : pd.DataFrame
        Must contain 'cell_probability', 'filter_rna_emptyDrops', 'filter_rna_max_mito'.
    ax : matplotlib.axes.Axes
    bins : int, optional
        Number of histogram bins. Default is 20.

    Returns
    -------
    matplotlib.axes.Axes
    """
    sns.histplot(x='cell_probability', data=metrics[(metrics.filter_rna_emptyDrops) & (metrics.filter_rna_max_mito)], ax=ax, bins=20)
    ax.set_xlabel('Cellbender cell prob.\nfor cells by EmptyDrops and mito. thresholds')
    return ax

# functions to plot ATAC QC
def rna_umis_vs_atac_hqaa_plot(metrics, ax):
    """
    Scatter plot of RNA UMIs vs. ATAC high-quality aligned reads.

    Parameters
    ----------
    metrics : pd.DataFrame
        Must contain 'rna_umis', 'atac_hqaa', 'pass_all_filters'.
    ax : matplotlib.axes.Axes
    alpha : float, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
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


def barcode_rank_plot_atac(metrics, ax, hue='pass_all_filters', alpha=0.02, s=3):
    """
    Create a barcode rank plot for ATAC high-quality aligned reads.

    Parameters
    ----------
    metrics : pd.DataFrame
        Must contain 'atac_hqaa' and the column specified by `hue`.
    ax : matplotlib.axes.Axes
    hue : str, optional
        Column name for color grouping.
    alpha : float, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = metrics.sort_values('atac_hqaa', ascending=False)
    df['barcode_rank'] = range(1, len(df) + 1)
    sns.scatterplot(x='barcode_rank', y='atac_hqaa', data=df, ax=ax, hue=hue, palette={True: 'red', False: 'black'}, edgecolor=None, alpha=alpha, s=s)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Barcode rank')
    ax.set_ylabel('Pass filter reads (ATAC)')
    return ax

def atac_hqaa_vs_atac_tss_enrichment_plot(metrics, ax, hue='pass_all_filters', alpha=0.02, s=3):
    """
    Scatter plot of ATAC high-quality reads vs. TSS enrichment.

    Parameters
    ----------
    metrics : pd.DataFrame
        Must contain 'atac_hqaa', 'atac_tss_enrichment', and the `hue` column.
    ax : matplotlib.axes.Axes
    hue : str, optional
    alpha : float, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    sns.scatterplot(x='atac_hqaa', y='atac_tss_enrichment', data=metrics, ax=ax, hue=hue, palette={True: 'red', False: 'black'}, edgecolor=None, alpha=alpha, s=s)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Pass filter reads (ATAC)')
    ax.set_ylabel('TSS enrichment')
    return ax

def atac_hqaa_vs_atac_mt_pct_plot(metrics, ax, hue='pass_all_filters', alpha=0.2, s=3):
    """
    Scatter plot of ATAC high-quality reads vs. mitochondrial percentage.

    Parameters
    ----------
    metrics : pd.DataFrame
        Must contain 'atac_hqaa', 'atac_percent_mitochondrial', and the `hue` column.
    ax : matplotlib.axes.Axes
    hue : str, optional
    alpha : float, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    sns.scatterplot(x='atac_hqaa', y='atac_percent_mitochondrial', data=metrics, ax=ax, hue=hue, palette={True: 'red', False: 'black'}, edgecolor=None, alpha=alpha, s=s)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlabel('Pass filter reads (ATAC)')
    ax.set_ylabel('atac_percent_mitochondrial')
    return ax

def atac_tss_enrichment_vs_atac_mt_pct_plot(metrics, ax, hue='pass_all_filters', alpha=0.2, s=3):
    """
    Scatter plot of ATAC TSS enrichment vs. mitochondrial percentage.

    Parameters
    ----------
    metrics : pd.DataFrame
        Must contain 'atac_tss_enrichment', 'atac_percent_mitochondrial',
        and the `hue` column.
    ax : matplotlib.axes.Axes
    hue : str, optional
    alpha : float, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    sns.scatterplot(x='atac_tss_enrichment', y='atac_percent_mitochondrial', data=metrics, ax=ax, hue=hue, palette={True: 'red', False: 'black'}, edgecolor=None, alpha=alpha, s=s)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('tss_enrichment')
    ax.set_ylabel('Fraction mito. (ATAC)')
    return ax


