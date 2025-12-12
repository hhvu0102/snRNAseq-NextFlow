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

from helper_joint_qc import *

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

# ---upfront thresholds--- 
THRESHOLD_CELLBENDER_MIN_CELL_PROBABILITY = 0.99
THRESHOLD_ATAC_MIN_TSS_ENRICHMENT = 2

# ---process inputs---
## ATAC --> RNA barcode mappings
rna_barcodes = pd.read_csv(RNA_BARCODE_WHITELIST, header=None)[0].to_list()
atac_barcodes = pd.read_csv(ATAC_BARCODE_WHITELIST, header=None)[0].to_list()
atac_to_rna = dict(zip(atac_barcodes, rna_barcodes))

## load metrics df
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
metrics['filter_cellbender_cell_probability'] = metrics.cell_probability >= THRESHOLD_CELLBENDER_MIN_CELL_PROBABILITY

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
MAX_EXPECTED_NUMBER_NUCLEI = round_up(len(metrics[metrics.rna_umis >= inflection]), 3)
LOWERBOUNDS = np.concatenate(([1, 5], np.arange(10, 251, 10), [300, 350, 400, 450, 500]))

for i in LOWERBOUNDS:
    UMI_THRESHOLD = estimate_threshold(metrics[(metrics.barcode!='-') & (metrics.rna_umis>=i)].rna_umis.astype(int))
    NUMBER_MEETING_UMI_THRESHOLD = (metrics.rna_umis>=UMI_THRESHOLD).sum()
    #allow 1% wiggle room
    if (NUMBER_MEETING_UMI_THRESHOLD*101/100 <= MAX_EXPECTED_NUMBER_NUCLEI) or (NUMBER_MEETING_UMI_THRESHOLD*99/100 <= MAX_EXPECTED_NUMBER_NUCLEI):
        break

if (NUMBER_MEETING_UMI_THRESHOLD*101/100 > MAX_EXPECTED_NUMBER_NUCLEI) and (NUMBER_MEETING_UMI_THRESHOLD*99/100 > MAX_EXPECTED_NUMBER_NUCLEI):
    # just fall back to 500
    UMI_THRESHOLD = 500
    NUMBER_MEETING_UMI_THRESHOLD = (metrics.rna_umis>=UMI_THRESHOLD).sum()

THRESHOLD_RNA_MIN_UMI = UMI_THRESHOLD
metrics['filter_rna_min_umi'] = metrics.rna_umis >= THRESHOLD_RNA_MIN_UMI

# get %ambient vs post CB UMI thresholding
peaks_cb, n_peaks_cb, cb_kde_df = guess_n_classes_cellbender(metrics)
THRESHOLD_FRACTION_CB_REMOVED, THRESHOLD_POST_CB_UMIS = get_cellbender_thresholds(metrics, peaks_cb, n_peaks_cb, cb_kde_df)

metrics['filter_pct_cellbender_removed'] = metrics.pct_cellbender_removed <= THRESHOLD_FRACTION_CB_REMOVED*100


### get THRESHOLD_EXON_GENE_BODY_RATIO
import skimage as ski
from scipy import ndimage as ndi
x = np.log10(metrics[(metrics.rna_exon_to_full_gene_body_ratio>0)&
                     (metrics.filter_rna_min_umi ==True)].rna_umis)
y = metrics[(metrics.rna_exon_to_full_gene_body_ratio>0)&
            (metrics.filter_rna_min_umi ==True)].rna_exon_to_full_gene_body_ratio

THRESHOLD_EXON_GENE_BODY_RATIO = get_exon_fullgene_ratio(x, y)

if THRESHOLD_EXON_GENE_BODY_RATIO >= 0.95:
    data = metrics[(metrics.rna_exon_to_full_gene_body_ratio>0)&
              (metrics.rna_exon_to_full_gene_body_ratio<1.0)].rna_exon_to_full_gene_body_ratio.astype(float).values
    THRESHOLD_EXON_GENE_BODY_RATIO = threshold_multiotsu(data, classes=3)[1]

### get THRESHOLD_RNA_MAX_MITO
n_peaks, rna_kde_df = guess_n_classes(metrics, "RNA")
THRESHOLD_RNA_MAX_MITO = get_chrMT_threshold_RNA(metrics, n_peaks = n_peaks)
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

# get HQAA threshold
values = np.log10(atac_metrics[(atac_metrics.tss_enrichment > 2)].hqaa).values
values = values.reshape((len(values),1))
thresholds = threshold_multiotsu(image=values, classes=2, nbins=256)
# convert back to linear scale
thresholds = [pow(10, i) for i in thresholds]
lower_thres = round(thresholds[0])
lower_thres = max(lower_thres, 100)
values = np.log10(atac_metrics[(atac_metrics.hqaa > lower_thres)].hqaa).values
values = values.reshape((len(values),1))
thresholds = threshold_multiotsu(image=values, classes=3, nbins=256)
# convert back to linear scale
thresholds = [pow(10, i) for i in thresholds]
THRESHOLD_ATAC_MIN_HQAA = round(thresholds[1])

metrics['filter_atac_min_hqaa'] = metrics.atac_hqaa >= THRESHOLD_ATAC_MIN_HQAA

### get THRESHOLD_ATAC_MAX_MITO
n_peaks, atac_kde_df = guess_n_classes(metrics, "ATAC")
THRESHOLD_ATAC_MAX_MITO = get_chrMT_threshold_ATAC(metrics, n_peaks = n_peaks)



### get cells that passed all thresholds; those that passed post-CB nUMIs have been identified above
metrics['filter_cellbender_cell_probability'] = metrics.cell_probability >= THRESHOLD_CELLBENDER_MIN_CELL_PROBABILITY
metrics['filter_rna_max_mito'] = metrics.rna_percent_mitochondrial <= THRESHOLD_RNA_MAX_MITO
metrics['filter_rna_exon_to_full_gene_body_ratio'] = metrics.rna_exon_to_full_gene_body_ratio <= THRESHOLD_EXON_GENE_BODY_RATIO
metrics['filter_atac_min_hqaa'] = metrics.atac_hqaa >= THRESHOLD_ATAC_MIN_HQAA
metrics['filter_atac_min_tss_enrichment'] = metrics.atac_tss_enrichment >= THRESHOLD_ATAC_MIN_TSS_ENRICHMENT
metrics['filter_atac_max_mito'] = metrics.atac_percent_mitochondrial <= THRESHOLD_ATAC_MAX_MITO
metrics['pass_all_filters'] = metrics.filter(like='filter_').all(axis=1)

# to collect all Thresholds here
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


# Plot QC metrics #to work on plotting
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(4*4, 3*4))

ax = axs[0, 0]
barcode_rank_plot(metrics, ax)
ax.axhline(knee, color='red', ls='--', label='knee={:,}'.format(knee))
ax.axhline(inflection, color='green', ls='--', label='inflection={:,}'.format(inflection))
ax.axhline(endCliff, color='blue', ls='--', label='end_cliff={:,}'.format(endCliff))
ax.axhline(plateau, color='orange', ls='--', label='plateau={:,}'.format(plateau))
ax.legend()

ax = axs[0, 1]
rna_umis_vs_rna_mito_plot(metrics, ax)
ax.axhline(THRESHOLD_RNA_MAX_MITO/100, color='blue', ls='--', label='THRESHOLD_RNA_MAX_MITO = {:,}'.format(THRESHOLD_RNA_MAX_MITO))
ax.axvline(THRESHOLD_RNA_MIN_UMI, color='red', ls='--')
ax.legend()

ax = axs[0, 2]
cellbender_fraction_removed(metrics, ax)
ax.axhline(THRESHOLD_FRACTION_CB_REMOVED, color='blue', ls='--')

ax = axs[0, 3]
sns.histplot(x='pct_cellbender_removed', data=metrics[(metrics.pct_cellbender_removed > 5) &
                                                      (metrics.pct_cellbender_removed < 50) &
                                                      (np.isnan(metrics.pct_cellbender_removed) == False)], ax=ax)
ax.axvline(THRESHOLD_FRACTION_CB_REMOVED*100, color='blue', ls='--', label='%ambient removed threshold Multi-otsu= {:,}'.format(round(THRESHOLD_FRACTION_CB_REMOVED*100, 2)))
ax.legend()
ax.set_xlabel('5% < % ambient removed < 50%')

ax = axs[1, 0]
cellbender_cell_probabilities(metrics, ax)

ax = axs[1, 1]
sns.scatterplot(x='barcode_rank', y='rna_umis', data=df[(df.barcode!='-') & (df.rna_umis > 5)], ax=ax, edgecolor=None, alpha=0.5, s=2, hue='range')
ax.axhline(endCliff, color='blue', ls='--', label='end_cliff = {:,}'.format(endCliff))
ax.axhline(knee, color='red', ls='--', label='knee = {:,}'.format(knee))
ax.axhline(inflection, color='green', ls='--', label='inflection = {:,}'.format(inflection))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('barcode rank')
ax.set_ylabel('UMIs > 5')

ax = axs[1, 2]
sns.scatterplot(x='x_new', y='change_hat', data=df_peak[df_peak.y_new > 5], ax=ax, edgecolor=None, alpha=0.5, s=3, hue='range')
ax.scatter(x=df_peak.x_new[final_peaks], y=df_peak.change_hat[final_peaks], color='red', s=10, zorder=5)  # Red dots
ax.set_xscale('log')
ax.set_xlabel('rank of barcodes with UMIs > 5')
ax.set_ylabel('Discrete diff. after smooth')

ax = axs[1, 3]
rna_umis_vs_exon_to_full_gene_body_ratio(metrics, ax)
ax.axhline(THRESHOLD_EXON_GENE_BODY_RATIO, color='red', ls='--', label='exon/full ratio. Multi-otsu = {:,}'.format(round(THRESHOLD_EXON_GENE_BODY_RATIO, 2)))
ax.legend()
ax.axvline(THRESHOLD_RNA_MIN_UMI, color='red', ls='--')
ax.set_xlim(left=0.8*THRESHOLD_RNA_MIN_UMI)

ax = axs[2, 0]
rna_umis_vs_atac_hqaa_plot(metrics, ax)
ax.axhline(THRESHOLD_ATAC_MIN_HQAA, color='red', ls='--')
ax.axvline(THRESHOLD_RNA_MIN_UMI, color='red', ls='--')

ax = axs[2, 1]
atac_hqaa_vs_atac_tss_enrichment_plot(metrics, ax, alpha=0.02)
ax.axvline(THRESHOLD_ATAC_MIN_HQAA, color='red', ls='--', label='THRESHOLD_ATAC_MIN_HQAA = {:,}'.format(THRESHOLD_ATAC_MIN_HQAA))
ax.axhline(THRESHOLD_ATAC_MIN_TSS_ENRICHMENT, color='red', ls='--')
ax.legend()

ax = axs[2, 2]
barcode_rank_plot_atac(metrics, ax, alpha=0.02)
ax.axhline(THRESHOLD_ATAC_MIN_HQAA, color='red', ls='--')

ax = axs[2, 3]
atac_hqaa_vs_atac_mt_pct_plot(metrics, ax, alpha=0.02)
ax.axvline(THRESHOLD_ATAC_MIN_HQAA, color='red', ls='--')
ax.axhline(THRESHOLD_ATAC_MAX_MITO, color='green', ls='--', label='THRESHOLD_ATAC_MAX_MITO = {:,}'.format(THRESHOLD_ATAC_MAX_MITO))
ax.legend()

#ax = axs[3,0]
#atac_tss_enrichment_vs_atac_mt_pct_plot(metrics, ax, alpha=0.02)
#ax.axvline(THRESHOLD_ATAC_MIN_TSS_ENRICHMENT, color='red', ls='--')

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

