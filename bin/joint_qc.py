#!/usr/bin/env python3
# coding: utf-8

import sys
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

import argparse

parser = argparse.ArgumentParser("Plot QC metrics per sample")
parser.add_argument("--sample", help="Donor ID.", type=str)
parser.add_argument("--RNA_results_dir", help="Path to RNA results directory.", type=str)
parser.add_argument("--ATAC_results_dir", help="Path to ATAC results directory.", type=str)
parser.add_argument("--RNA_BARCODE_WHITELIST", help="Path to RNA barcode whitelist.", type=str)
parser.add_argument("--ATAC_BARCODE_WHITELIST", help="Path to ATAC barcode whitelist.", type=str)
parser.add_argument("--qcPlot", help="Path to save qcPlot plots.", type=str)
parser.add_argument("--upsetPlot", help="Path to save upset plots.", type=str)
parser.add_argument("--outmetrics", help="Path to save all metrics results.", type=str)


args = parser.parse_args()

# ---inputs---
donor = args.sample
print(donor)
RNA_results_dir = str(args.RNA_results_dir)
print(RNA_results_dir)
ATAC_results_dir = args.ATAC_results_dir
RNA_BARCODE_WHITELIST = args.RNA_BARCODE_WHITELIST
ATAC_BARCODE_WHITELIST = args.ATAC_BARCODE_WHITELIST

CELLBENDER = RNA_results_dir+'cellbender/'+donor+'-hg38.cellbender_FPR_0.05.h5'

RNA_METRICS = RNA_results_dir+'qc/'+donor+'-hg38.qc.txt'
ATAC_METRICS = ATAC_results_dir+'ataqv/single-nucleus/'+donor+'-hg38.txt'
GENE_FULL_EXON_OVER_INTRON_COUNTS = RNA_results_dir + 'starsolo/' + donor + '-hg38/' + donor + '-hg38.Solo.out/GeneFull_ExonOverIntron/raw'
GENE_COUNTS = RNA_results_dir + 'starsolo/' + donor + '-hg38/' + donor + '-hg38.Solo.out/Gene/raw'
knee = RNA_results_dir + 'emptyDrops/' + donor + '-hg38.knee.txt'
passQC = RNA_results_dir + 'emptyDrops/' + donor + '-hg38.pass.txt'
atac_sugg = pd.read_csv(ATAC_results_dir + 'ataqv/single-nucleus/' + donor + '-hg38.suggested-thresholds.tsv', sep = '\t')
rna_sugg = pd.read_csv(RNA_results_dir + 'qc/' + donor + '-hg38.suggested-thresholds.tsv', sep = '\t')

# ---process inputs---
THRESHOLD_CELLBENDER_MIN_CELL_PROBABILITY = 0.99
THRESHOLD_ATAC_MIN_HQAA = atac_sugg.iloc[0]['threshold']
THRESHOLD_ATAC_MIN_TSS_ENRICHMENT = 2

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

# Multi-Otsu function
from skimage.filters import threshold_multiotsu
def estimate_threshold(x, classes=3):
    # do on logscale
    values = np.log10(x).values
    values = values.reshape((len(values),1))
    thresholds = threshold_multiotsu(image=values, classes=classes, nbins=256)
    # convert back to linear scale
    thresholds = [pow(10, i) for i in thresholds]
    UMI_THRESHOLD = round(thresholds[classes - 2])
    return UMI_THRESHOLD

# ATAC --> RNA barcode mappings
rna_barcodes = pd.read_csv(RNA_BARCODE_WHITELIST, header=None)[0].to_list()
atac_barcodes = pd.read_csv(ATAC_BARCODE_WHITELIST, header=None)[0].to_list()
atac_to_rna = dict(zip(atac_barcodes, rna_barcodes))

#load metrics df
adata = anndata_from_h5(CELLBENDER, analyzed_barcodes_only=True)
rna_metrics = pd.read_csv(RNA_METRICS, sep='\t')
rna_metrics = rna_metrics[rna_metrics.barcode!='-']

## Calculate ratio of exonic vs full gene body reads
# exons only
gene_mat = mmread(os.path.join(GENE_COUNTS, 'matrix.mtx'))
gene_umis_per_barcode = gene_mat.sum(axis=0).tolist()[0]

# includes introns
gene_full_mat = mmread(os.path.join(GENE_FULL_EXON_OVER_INTRON_COUNTS, 'matrix.mtx'))
gene_full_umis_per_barcode = gene_full_mat.sum(axis=0).tolist()[0]

barcodes = pd.read_csv(os.path.join(GENE_COUNTS, 'barcodes.tsv'), header=None)[0]
assert(all(barcodes == pd.read_csv(os.path.join(GENE_FULL_EXON_OVER_INTRON_COUNTS, 'barcodes.tsv'), header=None)[0]))

exon_to_full_gene_body_ratio = pd.DataFrame({'barcode': barcodes, 'gene': gene_umis_per_barcode, 'gene_full': gene_full_umis_per_barcode})
exon_to_full_gene_body_ratio['exon_to_full_gene_body_ratio'] = exon_to_full_gene_body_ratio.gene / exon_to_full_gene_body_ratio.gene_full
umis_genefull_exon_over_intron = exon_to_full_gene_body_ratio.set_index('barcode').gene_full.to_dict()
rna_metrics = rna_metrics.merge(exon_to_full_gene_body_ratio)
metrics = rna_metrics.set_index('barcode').rename(columns=lambda x: 'rna_' + x)

## cellbender-related statistics
metrics = metrics.reset_index()
cell_probability = cellbender_anndata_to_cell_probability(adata)
post_cellbender_umis = umi_count_after_decontamination(adata)

metrics['cell_probability'] = metrics.barcode.map(lambda x: cell_probability[x] if x in cell_probability else np.nan)
metrics['post_cellbender_umis'] = metrics.barcode.map(lambda x: post_cellbender_umis[x] if x in post_cellbender_umis else np.nan)
metrics['fraction_cellbender_removed'] = (metrics.rna_umis - metrics.post_cellbender_umis) / metrics.rna_umis
metrics['rna_percent_mitochondrial'] = metrics.rna_fraction_mitochondrial * 100
metrics['pct_cellbender_removed'] = metrics.fraction_cellbender_removed * 100

### get bc that passed emptydrops analysis
bc = pd.read_csv(passQC, header=0, delim_whitespace="\t") 
metrics['filter_rna_emptyDrops'] = metrics['barcode'].isin(bc.barcode)

### load metrics on knee plot
KNEE_FILE = knee
with open(KNEE_FILE, 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    next(reader, None)
    for row in reader:
        knee = round(float(row[0]))
        inflection = round(float(row[1]))
        inflection_rank = round(float(row[2]))
        knee_rank = round(float(row[3]))
        endCliff = round(float(row[4]))
        end_cliff_rank = round(float(row[5]))
        plateau = round(float(row[6]))

### get bc that passed threshold UMIs obtained using Multi-Otsu
# try to infer UMI threshold
import math
def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n / multiplier) * multiplier
MAX_EXPECTED_NUMBER_NUCLEI = round_up(len(metrics[metrics.rna_umis >= inflection]), 3)
LOWERBOUNDS = np.concatenate(([1, 5], np.arange(10, 251, 10), [300, 350, 400, 450, 500]))

for i in LOWERBOUNDS:
    UMI_THRESHOLD = estimate_threshold(metrics[(metrics.barcode!='-') & (metrics.rna_umis>=i)].rna_umis.astype(int))
    NUMBER_MEETING_UMI_THRESHOLD = (metrics.rna_umis>=UMI_THRESHOLD).sum()
    print(str(i) + ": " + str(UMI_THRESHOLD) + "; " + str(NUMBER_MEETING_UMI_THRESHOLD))
    #allow 1% wiggle room
    if (NUMBER_MEETING_UMI_THRESHOLD*101/100 <= MAX_EXPECTED_NUMBER_NUCLEI) or (NUMBER_MEETING_UMI_THRESHOLD*99/100 <= MAX_EXPECTED_NUMBER_NUCLEI):
        break

if (NUMBER_MEETING_UMI_THRESHOLD*101/100 > MAX_EXPECTED_NUMBER_NUCLEI) and (NUMBER_MEETING_UMI_THRESHOLD*99/100 > MAX_EXPECTED_NUMBER_NUCLEI):
    # just fall back to 500
    UMI_THRESHOLD = 500
    NUMBER_MEETING_UMI_THRESHOLD = (metrics.rna_umis>=UMI_THRESHOLD).sum()

THRESHOLD_RNA_MIN_UMI = UMI_THRESHOLD
metrics['filter_rna_min_umi'] = metrics.rna_umis >= THRESHOLD_RNA_MIN_UMI


### get THRESHOLD_POST_CB_UMIS
THRESHOLD_POST_CB_UMIS = estimate_threshold(metrics[(np.isnan(metrics.pct_cellbender_removed) == False) &
                                                  (metrics.post_cellbender_umis > 0)].post_cellbender_umis.astype(float),
                                                  classes = 2)
#metrics['filter_post_cellbender_umis'] = metrics.post_cellbender_umis >= THRESHOLD_POST_CB_UMIS #this will not work well when there is only one distribution of UMIs e.g. in sample HPAP-079

### get THRESHOLD_FRACTION_CB_REMOVED
THRESHOLD_FRACTION_CB_REMOVED = estimate_threshold(metrics[(metrics.filter_rna_emptyDrops == True) &
                                                           (metrics.filter_rna_min_umi == True) &
                                                           (metrics.pct_cellbender_removed>0) &
                                                           (metrics.pct_cellbender_removed<100) &
                                                           (np.isnan(metrics.pct_cellbender_removed) == False)].pct_cellbender_removed.astype(float),
                                                   classes = 4)
metrics['filter_pct_cellbender_removed'] = metrics.pct_cellbender_removed <= THRESHOLD_FRACTION_CB_REMOVED

### get THRESHOLD_EXON_GENE_BODY_RATIO
import skimage as ski
from scipy import ndimage as ndi
x = np.log10(metrics[(metrics.rna_exon_to_full_gene_body_ratio>0)&
                     (metrics.filter_rna_min_umi ==True)].rna_umis)
y = metrics[(metrics.rna_exon_to_full_gene_body_ratio>0)&
            (metrics.filter_rna_min_umi ==True)].rna_exon_to_full_gene_body_ratio

# Create a 2D array representation
heatmap, xedges, yedges = np.histogram2d(x, y, bins=50) # the smaller bins is, the smoother the heatmap would be. bins=150 was chosen after testing 50, 100, 150, 200 and 300

smooth = ski.filters.gaussian(heatmap, sigma=2) #use Gaussian filtering to smooth out the data points that do not cluster together
thresh = smooth > threshold_multiotsu(image=smooth, classes = 4)[1] #use Multi-Otsu to estimate a threshold that marks foreground and background in the image `smooth`
labels = ski.morphology.label(thresh)
labelCount = np.bincount(labels.ravel())
background = np.argmax(labelCount)
thresh[labels != background] = 255
heatmap_seg = thresh

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
    THRESHOLD_EXON_GENE_BODY_RATIO = max_y_coordinate
else:
    data = metrics[(metrics.rna_exon_to_full_gene_body_ratio>0)&
              (metrics.rna_exon_to_full_gene_body_ratio<1.0)].rna_exon_to_full_gene_body_ratio.astype(float).values
    THRESHOLD_EXON_GENE_BODY_RATIO = threshold_multiotsu(data, classes=3)[1]

if THRESHOLD_EXON_GENE_BODY_RATIO >= 0.95:
    data = metrics[(metrics.rna_exon_to_full_gene_body_ratio>0)&
              (metrics.rna_exon_to_full_gene_body_ratio<1.0)].rna_exon_to_full_gene_body_ratio.astype(float).values
    THRESHOLD_EXON_GENE_BODY_RATIO = threshold_multiotsu(data, classes=3)[1]

### get THRESHOLD_RNA_MAX_MITO
# Step 0: check the number of distributions along the RNA_mito_percent axis
from scipy.signal import find_peaks
## Step 0.1: guess lower bound to preclude
data = metrics[(metrics.filter_rna_emptyDrops == True) &
               (metrics.filter_rna_min_umi == True) &
               (metrics.rna_percent_mitochondrial > 1) &
               (metrics.rna_percent_mitochondrial < 100) &
               (metrics.filter_pct_cellbender_removed == True)].rna_percent_mitochondrial.astype(float)
values = np.log10(data).values
values = values.reshape((len(values),1))
thresholds = threshold_multiotsu(image=values, classes=4, nbins=256)
# convert back to linear scale
thresholds = [pow(10, i) for i in thresholds]

# Step 0.2: get density and check number of distributions
data = np.log10(metrics[(metrics.filter_rna_emptyDrops == True) &
                        (metrics.filter_rna_min_umi == True) &
                        (metrics.rna_percent_mitochondrial > thresholds[0]) &
                        (metrics.rna_percent_mitochondrial < 50) &
                        (metrics.filter_pct_cellbender_removed == True)].rna_percent_mitochondrial.astype(float))

# Generate KDE object from the data
kde = sns.kdeplot(data)
# The plotted data is stored in kde.lines[0].get_xdata() and .get_ydata()
x = kde.lines[0].get_xdata()
y = kde.lines[0].get_ydata()

peaks, _ = find_peaks(y, prominence=abs(max(y) * 0.05))
n_knee = len(peaks)

print('Number of prominent cliff RNA %chrMT is {:,}'.format(n_knee)) #if there's one distribution, use 2D matrix; if more than one, use 1D array

# Store in DataFrame
rna_kde_df = pd.DataFrame({'x': x, 'density': y})
# Clear the plot
import matplotlib.pyplot as plt
plt.clf()

# Step 1: get thresholds
if n_knee == 1:
    import skimage as ski
    from scipy import ndimage as ndi
    # Subset the nuclei to those that passed both emptydrops and post-CB nUMI thresholds, and have 1% < %chrMT < 50% to determine the %chrMT threshold. 50% is used since %chrMT per nucleus/cell should be below this threshold in practice https://pmc.ncbi.nlm.nih.gov/articles/PMC8599307/
    x = np.log10(metrics[(metrics.filter_rna_emptyDrops == True) & 
                        (metrics.filter_rna_min_umi == True) &
                        (metrics.rna_percent_mitochondrial > 0) &
                        (metrics.rna_percent_mitochondrial < 50) &
                        (metrics.filter_pct_cellbender_removed == True)].rna_umis)
    y = np.log10(metrics[(metrics.filter_rna_emptyDrops == True) & 
                (metrics.filter_rna_min_umi == True) &
                (metrics.rna_percent_mitochondrial > 0) &
                (metrics.rna_percent_mitochondrial < 50) &
                (metrics.filter_pct_cellbender_removed == True)].rna_percent_mitochondrial)

    # Create a 2D array representation
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=150) # the smaller bins is, the smoother the heatmap would be. bins=150 was chosen after testing 50, 100, 150, 200 and 300

    smooth = ski.filters.gaussian(heatmap, sigma=2) #use Gaussian filtering to smooth out the data points that do not cluster together
    thresh = smooth > threshold_multiotsu(image=smooth, classes = 4)[1] #use Multi-Otsu to estimate a threshold that marks foreground and background in the image `smooth`
    labels = ski.morphology.label(thresh)
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    thresh[labels != background] = 255
    heatmap_seg = thresh

    true_indices = np.argwhere(np.any(heatmap_seg.T, axis=1))
    if true_indices.size > 0:
        # Get the highest row index that contains True
        max_true_row_index = np.max(true_indices)
        # Convert this row index to the corresponding y-coordinate using extent
        extent_top = yedges[-1]
        extent_bottom = yedges[0]
        number_of_rows = heatmap_seg.shape[0]

        # Calculate the y-coordinate
        max_y_coordinate = extent_bottom + (extent_top - extent_bottom) * (max_true_row_index / (number_of_rows - 1))
    else:
        print("Not find a threshold in the RNA array yet; change to not taking log transform.")
        y = metrics[(metrics.filter_rna_emptyDrops == True) &
                         (metrics.filter_rna_min_umi == True) &
                         (metrics.rna_percent_mitochondrial > 0) &
                         (metrics.rna_percent_mitochondrial < 50) &
                         (metrics.filter_pct_cellbender_removed == True)].rna_percent_mitochondrial
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=150)
        smooth = ski.filters.gaussian(heatmap, sigma=2)
        thresh = smooth > threshold_multiotsu(image=smooth, classes = 4)[1] #use Multi-Otsu to estimate a threshold that marks foreground and background in the image `smooth`
        labels = ski.morphology.label(thresh)
        labelCount = np.bincount(labels.ravel())
        background = np.argmax(labelCount)
        thresh[labels != background] = 255
        heatmap_seg = thresh
        
        true_indices = np.argwhere(np.any(heatmap_seg.T, axis=1))
        if true_indices.size > 0:
            max_true_row_index = np.max(true_indices)
            extent_top = yedges[-1]
            extent_bottom = yedges[0]
            number_of_rows = heatmap_seg.shape[0]
            max_y_coordinate = extent_bottom + (extent_top - extent_bottom) * (max_true_row_index / (number_of_rows - 1))
            max_y_coordinate = np.log10(max_y_coordinate)
        else:
            print("Not find a threshold in the RNA array yet; fall back to Multi-Otsu on 1D array of chrMT.")
            max_y_coordinate = np.log10(estimate_threshold(metrics[(metrics.filter_rna_emptyDrops == True) &
                                                                 (metrics.rna_percent_mitochondrial > 1) &
                                                                 (metrics.rna_percent_mitochondrial < 50) &
                                                                 (metrics.filter_pct_cellbender_removed == True)].rna_percent_mitochondrial.astype(float), classes = n_knee+1))

    THRESHOLD_RNA_MAX_MITO = round(pow(10, max_y_coordinate))
else:
    THRESHOLD_RNA_MAX_MITO = estimate_threshold(metrics[(metrics.filter_rna_emptyDrops == True) &
                                                       (metrics.rna_percent_mitochondrial > 1) &
                                                       (metrics.rna_percent_mitochondrial < 50) &
                                                       (metrics.filter_pct_cellbender_removed == True)].rna_percent_mitochondrial.astype(float), classes = n_knee+1)
##############################

####### knee plot analysis
def get_color(umis): #to plot the intervals
    if umis < endCliff:
        return 'UMIs < ' + str(endCliff)
    elif endCliff <= umis < knee:
        return str(endCliff) + ' < UMIs < ' + str(knee)
    else:
        return 'UMIs > ' + str(knee)

df = metrics.sort_values('rna_umis', ascending=False)
df['barcode_rank'] = range(1, len(df) + 1)
df = df[df.rna_umis > 0] #to avoid taking log10(0)
df['range'] = df['rna_umis'].apply(get_color)

#using diff() to calculate the n-th order discrete difference between two consecutive data points
change = np.diff(np.log10(df.rna_umis).values) # / np.diff(np.log10(df.barcode_rank).values)
change = np.append([0], change)
df['change_umis'] = abs(change)

#interpolate to sample from data on a log scale and obtain data points that are equally spaced. This step is important to do smoothing (savgol_filter) later,
from scipy.interpolate import interp1d
f = interp1d(x = np.log10(df.barcode_rank), y = np.log10(df.rna_umis))
reg_t = np.linspace(start=np.log10(min(df.barcode_rank)), stop=np.log10(max(df.barcode_rank)), num=int(len(df.barcode_rank)*2))
reg = f(reg_t)
df2 = pd.DataFrame(columns=['x_new', 'y_new'])
df2['x_new'] = 10**reg_t
df2['y_new'] = 10**reg
df2['range'] = df2['y_new'].apply(get_color)

#do savgol_filter, which essentially smooths out the data and helps to focus on the slopes (degree of change) only
from scipy.signal import savgol_filter
if inflection_rank >= knee_rank:
    filtered_df = df2[(df2['x_new'] >= knee_rank) & (df2['x_new'] <= inflection_rank)]
    w = filtered_df.shape[0] # Count how many interpolated points fall into [knee_rank, inflection_rank]
else:
    filtered_df = df2[(df2['x_new'] >= inflection_rank) & (df2['x_new'] <= knee_rank)] ## technically inflection_rank should always be higehr than knee_rank, but inflection in emptyDrops does not have smoothing so it's very unstable and one rank (i.e., x) can lead to multiple inflection point (but y, i.e. the UMI number, is still the same). In case inflection_rank is < knee_rank, force the higher point to be inflection_rank
    w = filtered_df.shape[0] # Count how many interpolated points fall into [knee_rank, inflection_rank]

w = round(w/5)
if w % 2 == 0:
    w += 1

w = max(w, 201) #if w is too small, it does not filter enough noise, hence force it to be 201 when w is too small as is

print("window size:")
print(w)

yhat = savgol_filter(x = np.log10(df2.y_new).values, window_length = w, polyorder = 1)
yhat_change = np.diff(yhat) 
yhat_change = np.append([0], yhat_change)
df2['change'] = yhat_change # using yhat_change instead of abs(yhat_change) to keep direction of changes

# make knee plot warning:
from scipy.signal import find_peaks

window_length = round(w*2)
if window_length % 2 == 0:
    window_length += 1

df_peak = pd.DataFrame(columns=['x_new', 'y_new', 'change'])
df_peak = df2
yhat = savgol_filter(x = df_peak.change, window_length = window_length, polyorder = 1) #has to smooth it out first
df_peak['change_hat'] = yhat

x = abs(df_peak[df_peak.y_new > 5].change_hat)
peaks, _ = find_peaks(x, prominence=abs(min(df_peak.change_hat) * 0.1))
n_knee = len(peaks)

if n_knee > 1 and (df_peak.x_new[peaks] > end_cliff_rank).sum() > 1:
    # the changes at the tail end of the knee plot can get unstable, so it has multiple peaks at times
    # in that case, keep the highest peak
    rhs_peaks = peaks[df_peak.x_new[peaks] > end_cliff_rank]
    rhs_kept_peak = df_peak.change_hat[rhs_peaks].idxmin()
    final_peaks = np.append(peaks[df_peak.x_new[peaks] < end_cliff_rank], rhs_kept_peak)
else:
    final_peaks = peaks

n_knee = len(final_peaks)

if n_knee == 1 and (df_peak.x_new[peaks] > end_cliff_rank).all():
    print("Warning: Knee plot does not show clear knee points")
elif n_knee == 0:
    print("Error: Could not detect peaks of change? Check data")
elif n_knee > 2:
    print("Warning: Knee plot may have multiple knees")
#elif (df_peak.x_new[peaks] < knee_rank).any():
#    print("Warning: There may be multiple knees in the plot")

print('Number of prominent cliff in knee plot analysis is {:,}'.format(n_knee))
##############################
##############################

### ATAC side ###
atac_metrics = pd.read_csv(ATAC_METRICS, sep='\t', index_col=0).rename_axis(index='barcode')
KEEP_ATAC_METRICS = ['median_fragment_length', 'hqaa', 'max_fraction_reads_from_single_autosome', 'percent_mitochondrial', 'tss_enrichment']
atac_metrics = atac_metrics[KEEP_ATAC_METRICS]
atac_metrics.max_fraction_reads_from_single_autosome = atac_metrics.max_fraction_reads_from_single_autosome.fillna(0)
atac_metrics.median_fragment_length = atac_metrics.median_fragment_length.fillna(0)
atac_metrics.percent_mitochondrial = atac_metrics.percent_mitochondrial.fillna(0)
atac_metrics.tss_enrichment = atac_metrics.tss_enrichment.fillna(0)
atac_metrics['fraction_mitochondrial'] = atac_metrics.percent_mitochondrial / 100

atac_metrics.index = atac_metrics.index.map(atac_to_rna)

metrics = metrics.set_index('barcode').rename(columns=lambda x: '' + x).join(atac_metrics.rename(columns=lambda x: 'atac_' + x))

# %chrMT on ATAC module
metrics['filter_atac_min_hqaa'] = metrics.atac_hqaa >= THRESHOLD_ATAC_MIN_HQAA

### get THRESHOLD_ATAC_MAX_MITO
# Step 0: check the number of distributions along the RNA_mito_percent axis
from scipy.signal import find_peaks
data = np.log10(metrics[(metrics.filter_atac_min_hqaa == True) &
                      (metrics.atac_percent_mitochondrial > 1) &
                      (metrics.atac_percent_mitochondrial < 50)].atac_percent_mitochondrial.astype(float))

# Generate KDE object from the data
kde = sns.kdeplot(data)
# The plotted data is stored in kde.lines[0].get_xdata() and .get_ydata()
x = kde.lines[0].get_xdata()
y = kde.lines[0].get_ydata()

peaks, _ = find_peaks(y, prominence=abs(max(y) * 0.05))
n_knee = len(peaks)

print('Number of prominent cliff ATAC %chrMT is {:,}'.format(n_knee)) #if there's one distribution, use 2D matrix; if more than one, use 1D array

# Store in DataFrame
atac_kde_df = pd.DataFrame({'x': x, 'density': y})
# Clear the plot
import matplotlib.pyplot as plt
plt.clf()

# Step 1: get thresholds
if n_knee == 1:
    import skimage as ski
    from scipy import ndimage as ndi
    # Subset the nuclei to those that passed both emptydrops and post-CB nUMI thresholds, and have 1% < %chrMT < 50% to determine the %chrMT threshold. 50% is used since %chrMT per nucleus/cell should be below this threshold in practice https://pmc.ncbi.nlm.nih.gov/articles/PMC8599307/
    x = np.log10(metrics[(metrics.filter_atac_min_hqaa == True) &
                        (metrics.atac_percent_mitochondrial > 0) &
                        (metrics.atac_percent_mitochondrial < 50)].atac_hqaa)
    y = np.log10(metrics[(metrics.filter_atac_min_hqaa == True) &
                        (metrics.atac_percent_mitochondrial > 0) &
                        (metrics.atac_percent_mitochondrial < 50)].atac_percent_mitochondrial)

    # Create a 2D array representation
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=150) # the smaller bins is, the smoother the heatmap would be. bins=150 was chosen after testing 50, 100, 150, 200 and 300

    smooth = ski.filters.gaussian(heatmap, sigma=2) #use Gaussian filtering to smooth out the data points that do not cluster together
    thresh = smooth > threshold_multiotsu(image=smooth, classes = 4)[1] #use Multi-Otsu to estimate a threshold that marks foreground and background in the image `smooth`
    labels = ski.morphology.label(thresh)
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    thresh[labels != background] = 255
    heatmap_seg = thresh
    true_indices = np.argwhere(np.any(heatmap_seg.T, axis=1))
    
    if true_indices.size > 0:
        # Get the highest row index that contains True
        max_true_row_index = np.max(true_indices)
        # Convert this row index to the corresponding y-coordinate using extent
        extent_top = yedges[-1]
        extent_bottom = yedges[0]
        number_of_rows = heatmap_seg.shape[0]

        # Calculate the y-coordinate
        max_y_coordinate = extent_bottom + (extent_top - extent_bottom) * (max_true_row_index / (number_of_rows - 1))
    else:
        print("Not find a threshold in the ATAC array yet; change to not taking log transform.")
        y = metrics[(metrics.filter_atac_min_hqaa == True) &
                        (metrics.atac_percent_mitochondrial > 0) &
                        (metrics.atac_percent_mitochondrial < 50)].atac_percent_mitochondrial
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=150) 
        smooth = ski.filters.gaussian(heatmap, sigma=2) #use Gaussian filtering to smooth out the data points that do not cluster together
        thresh = smooth > threshold_multiotsu(image=smooth, classes = 4)[1] #use Multi-Otsu to estimate a threshold that marks foreground and background in the image `smooth`
        labels = ski.morphology.label(thresh)
        labelCount = np.bincount(labels.ravel())
        background = np.argmax(labelCount)
        thresh[labels != background] = 255
        heatmap_seg = thresh        
        true_indices = np.argwhere(np.any(heatmap_seg.T, axis=1))
        if true_indices.size > 0:
            max_true_row_index = np.max(true_indices)
            max_true_row_index = np.max(true_indices)
            extent_top = yedges[-1]
            extent_bottom = yedges[0]
            number_of_rows = heatmap_seg.shape[0]
            max_y_coordinate = extent_bottom + (extent_top - extent_bottom) * (max_true_row_index / (number_of_rows - 1))
            max_y_coordinate = np.log10(max_y_coordinate)
        else:
            print("Not find a threshold in the ATAC array yet; fall back to Multi-Otsu on 1D array of chrMT.")
            max_y_coordinate = np.log10(estimate_threshold(metrics[(metrics.filter_atac_min_hqaa == True) &
                                                        (metrics.atac_percent_mitochondrial > 1) &
                                                        (metrics.atac_percent_mitochondrial < 50)].atac_percent_mitochondrial.astype(float), classes = n_knee+1))

    THRESHOLD_ATAC_MAX_MITO = round(pow(10, max_y_coordinate))
    
else:
    THRESHOLD_ATAC_MAX_MITO = estimate_threshold(metrics[(metrics.filter_atac_min_hqaa == True) &
                                                        (metrics.atac_percent_mitochondrial > 1) &
                                                        (metrics.atac_percent_mitochondrial < 50)].atac_percent_mitochondrial.astype(float), classes = n_knee+1)


### get cells that passed all thresholds; those that passed post-CB nUMIs have been identified above
#metrics['filter_cellbender_cell_probability'] = metrics.cell_probability >= THRESHOLD_CELLBENDER_MIN_CELL_PROBABILITY
metrics['filter_rna_max_mito'] = metrics.rna_percent_mitochondrial <= THRESHOLD_RNA_MAX_MITO
metrics['filter_rna_exon_to_full_gene_body_ratio'] = metrics.rna_exon_to_full_gene_body_ratio <= THRESHOLD_EXON_GENE_BODY_RATIO
metrics['filter_atac_min_hqaa'] = metrics.atac_hqaa >= THRESHOLD_ATAC_MIN_HQAA
metrics['filter_atac_min_tss_enrichment'] = metrics.atac_tss_enrichment >= THRESHOLD_ATAC_MIN_TSS_ENRICHMENT
metrics['filter_atac_max_mito'] = metrics.atac_percent_mitochondrial <= THRESHOLD_ATAC_MAX_MITO
metrics['pass_all_filters'] = metrics.filter(like='filter_').all(axis=1)
min_umi = metrics[metrics['pass_all_filters']].rna_umis.min()
if np.isnan(min_umi):
    print("Error: No barcodes passed all filters")
    print("Using inflection point for plotting purpose")
    min_umi = inflection

print("THRESHOLD_RNA_MIN_UMI = {:,}".format(THRESHOLD_RNA_MIN_UMI))
print("THRESHOLD_FRACTION_CB_REMOVED = {:,}".format(THRESHOLD_FRACTION_CB_REMOVED))
print("THRESHOLD_RNA_MAX_MITO = {:,}".format(THRESHOLD_RNA_MAX_MITO))
print("THRESHOLD_EXON_GENE_BODY_RATIO = {:,}".format(THRESHOLD_EXON_GENE_BODY_RATIO))
print("THRESHOLD_ATAC_MIN_HQAA = {:,}".format(THRESHOLD_ATAC_MIN_HQAA))
print("THRESHOLD_ATAC_MIN_TSS_ENRICHMENT = {:,}".format(THRESHOLD_ATAC_MIN_TSS_ENRICHMENT))
print("THRESHOLD_ATAC_MAX_MITO = {:,}".format(THRESHOLD_ATAC_MAX_MITO))


##########
metrics = metrics.reset_index()
# List of pass-QC barcodes
pass_qc_nuclei = list(sorted(metrics[metrics.pass_all_filters].barcode.to_list()))


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
    ax.set_yscale('log')
    ax.set_xlabel('Pass filter reads (ATAC)')
    ax.set_ylabel('atac_percent_mitochondrial')
    return ax

def atac_tss_enrichment_vs_atac_mt_pct_plot(metrics, ax, hue='pass_all_filters', alpha=0.2):
    sns.scatterplot(x='atac_tss_enrichment', y='atac_percent_mitochondrial', data=metrics, ax=ax, hue=hue, palette={True: 'red', False: 'black'}, edgecolor=None, alpha=alpha, s=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('tss_enrichment')
    ax.set_ylabel('atac_percent_mitochondrial')
    return ax

# Plot QC metrics
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(4*4, 16))

ax = axs[0, 0]
barcode_rank_plot(metrics, ax)
ax.axhline(knee, color='red', ls='--', label='knee={:,}'.format(knee))
ax.axhline(inflection, color='green', ls='--', label='inflection={:,}'.format(inflection))
ax.axhline(endCliff, color='blue', ls='--', label='end_cliff={:,}'.format(endCliff))
ax.axhline(plateau, color='orange', ls='--', label='plateau={:,}'.format(plateau))
ax.legend()

ax = axs[0, 1]
rna_umis_vs_rna_mito_plot(metrics, ax)
ax.axhline(THRESHOLD_RNA_MAX_MITO/100, color='blue', ls='--')
ax.axvline(min_umi, color='red', ls='--')

ax = axs[0, 2] #this plot is subjected to changes when optimizing for %MT thresholds
ax.scatter(x=rna_kde_df.x, y=rna_kde_df.density, s=1, zorder=5)
ax.set_xlabel('log10[Percent mito. (RNA)]')
ax.set_ylabel('Density')
ax.axvline(np.log10(THRESHOLD_RNA_MAX_MITO), color='orange', ls='--', label='THRESHOLD_RNA_MAX_MITO_PCT = {:,}'.format(THRESHOLD_RNA_MAX_MITO))
ax.legend()

ax = axs[0, 3]
cellbender_fraction_removed(metrics, ax)

ax = axs[1, 0]
#sns.histplot(x='post_cellbender_umis', data=metrics, ax=ax, log_scale=True)
#ax.axvline(THRESHOLD_POST_CB_UMIS, color='blue', ls='--', label='THRESHOLD_POST_CB_UMIS Multi-otsu= {:,}'.format(THRESHOLD_POST_CB_UMIS))
#ax.legend()
#ax.set_xlabel('post_cellbender_umis')
sns.histplot(x='pct_cellbender_removed', data=metrics[(metrics.filter_rna_emptyDrops == True) &
                                                      (metrics.rna_umis >= THRESHOLD_RNA_MIN_UMI) &
                                                      (metrics.pct_cellbender_removed>0) &
                                                      (metrics.pct_cellbender_removed<100) &
                                                      (np.isnan(metrics.pct_cellbender_removed) == False)], ax=ax, log_scale=True)
ax.axvline(THRESHOLD_FRACTION_CB_REMOVED, color='blue', ls='--', label='%ambient removed threshold Multi-otsu= {:,}'.format(THRESHOLD_FRACTION_CB_REMOVED))
ax.legend()
ax.set_xlabel('% ambient removed')

ax = axs[1, 1]
cellbender_cell_probabilities(metrics, ax)

ax = axs[1, 2]
sns.scatterplot(x='barcode_rank', y='rna_umis', data=df[(df.barcode!='-') & (df.rna_umis > 5)], ax=ax, edgecolor=None, alpha=0.5, s=2, hue='range')
ax.axhline(endCliff, color='blue', ls='--', label='end_cliff = {:,}'.format(endCliff))
ax.axhline(knee, color='red', ls='--', label='knee = {:,}'.format(knee))
ax.axhline(inflection, color='green', ls='--', label='inflection = {:,}'.format(inflection))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('barcode rank')
ax.set_ylabel('UMIs > 5')

ax = axs[1, 3]
sns.scatterplot(x='x_new', y='change_hat', data=df_peak[df_peak.y_new > 5], ax=ax, edgecolor=None, alpha=0.5, s=3, hue='range')
ax.scatter(x=df_peak.x_new[final_peaks], y=df_peak.change_hat[final_peaks], color='red', s=10, zorder=5)  # Red dots
ax.set_xscale('log')
ax.set_xlabel('rank of barcodes with UMIs > 5')
ax.set_ylabel('Discrete diff. after smooth')

ax = axs[2, 0]
rna_umis_vs_exon_to_full_gene_body_ratio(metrics, ax)
ax.axhline(THRESHOLD_EXON_GENE_BODY_RATIO, color='red', ls='--')
ax.axvline(min_umi, color='red', ls='--')
ax.set_xlim(left=0.8*min_umi)

ax = axs[2, 1]
# Plot using seaborn's histplot
sns.histplot(x='rna_exon_to_full_gene_body_ratio', data=metrics[(metrics.rna_exon_to_full_gene_body_ratio > 0) & 
                                                               (metrics.rna_exon_to_full_gene_body_ratio < 1.0)],
    ax=ax, log_scale=True)
ax.axvline(THRESHOLD_EXON_GENE_BODY_RATIO, color='red', ls='--', 
    label='exon/full ratio. Multi-otsu = {:,}'.format(round(THRESHOLD_EXON_GENE_BODY_RATIO, 2)))
ax.legend()
ax.set_xlabel('exon vs. full gene body ratio')

ax = axs[2, 2]
rna_umis_vs_atac_hqaa_plot(metrics, ax)
ax.axhline(THRESHOLD_ATAC_MIN_HQAA, color='red', ls='--')
ax.axvline(inflection, color='red', ls='--')

ax = axs[2, 3]
atac_hqaa_vs_atac_tss_enrichment_plot(metrics, ax, alpha=0.02)
ax.axvline(THRESHOLD_ATAC_MIN_HQAA, color='red', ls='--', label='THRESHOLD_ATAC_MIN_HQAA = {:,}'.format(THRESHOLD_ATAC_MIN_HQAA))
ax.axhline(THRESHOLD_ATAC_MIN_TSS_ENRICHMENT, color='red', ls='--')
ax.legend()

ax = axs[3, 0]
ax.scatter(x=atac_kde_df.x, y=atac_kde_df.density, s=1, zorder=5)
ax.set_xlabel('log10[Percent mito. (ATAC)]')
ax.set_ylabel('Density')
ax.axvline(np.log10(THRESHOLD_ATAC_MAX_MITO), color='orange', ls='--', label='THRESHOLD_ATAC_MAX_MITO_PCT = {:,}'.format(THRESHOLD_ATAC_MAX_MITO))
ax.legend()

ax = axs[3, 1]
barcode_rank_plot_atac(metrics, ax, alpha=0.02)
ax.axhline(THRESHOLD_ATAC_MIN_HQAA, color='red', ls='--')

ax = axs[3, 2]
atac_hqaa_vs_atac_mt_pct_plot(metrics, ax, alpha=0.02)
ax.axvline(THRESHOLD_ATAC_MIN_HQAA, color='red', ls='--')
ax.axhline(THRESHOLD_ATAC_MAX_MITO, color='green', ls='--')

ax = axs[3, 3]
atac_tss_enrichment_vs_atac_mt_pct_plot(metrics, ax, alpha=0.02)
ax.axvline(THRESHOLD_ATAC_MIN_TSS_ENRICHMENT, color='red', ls='--')

fig.suptitle('{:,} pass QC nuclei'.format(len(pass_qc_nuclei)) + " " + donor)
fig.tight_layout()
fig.savefig(args.qcPlot, bbox_inches='tight', dpi=300)

# Plot the number of cells passing each filter
fig, ax = plt.subplots(figsize=(7, 6))
ax.remove()

for_upset = metrics.filter(like='filter_').rename(columns=lambda x: 'pass_' + x)
for_upset = for_upset.groupby(for_upset.columns.to_list()).size()
upsetplot.plot(for_upset, fig=fig, sort_by='cardinality', show_counts=True)
fig.savefig(args.upsetPlot, bbox_inches='tight', dpi=300)


metrics.to_csv(args.outmetrics, index=False) 

