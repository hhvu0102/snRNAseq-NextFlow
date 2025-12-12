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
parser.add_argument("--ATAC_results_dir", help="Path to ATAC results directory.", type=str)
parser.add_argument("--qcPlot", help="Path to save qcPlot plots.", type=str)
parser.add_argument("--upsetPlot", help="Path to save upset plots.", type=str)
parser.add_argument("--outmetrics", help="Path to save all metrics results.", type=str)


args = parser.parse_args()

# ---inputs---
donor = args.sample
print(donor)
ATAC_results_dir = args.ATAC_results_dir

# ---upfront thresholds--- 
THRESHOLD_ATAC_MIN_TSS_ENRICHMENT = 2


### ATAC side ###
atac_metrics = pd.read_csv(ATAC_METRICS, sep='\t', index_col=0).rename_axis(index='barcode')
KEEP_ATAC_METRICS = ['median_fragment_length', 'hqaa', 'max_fraction_reads_from_single_autosome', 'percent_mitochondrial', 'tss_enrichment']
atac_metrics = atac_metrics[KEEP_ATAC_METRICS]
atac_metrics.max_fraction_reads_from_single_autosome = atac_metrics.max_fraction_reads_from_single_autosome.fillna(0)
atac_metrics.median_fragment_length = atac_metrics.median_fragment_length.fillna(0)
atac_metrics.percent_mitochondrial = atac_metrics.percent_mitochondrial.fillna(0)
atac_metrics.tss_enrichment = atac_metrics.tss_enrichment.fillna(0)
atac_metrics['fraction_mitochondrial'] = atac_metrics.percent_mitochondrial / 100

metrics = atac_metrics.rename(columns=lambda x: 'atac_' + x)

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
metrics['filter_atac_min_hqaa'] = metrics.atac_hqaa >= THRESHOLD_ATAC_MIN_HQAA
metrics['filter_atac_min_tss_enrichment'] = metrics.atac_tss_enrichment >= THRESHOLD_ATAC_MIN_TSS_ENRICHMENT
metrics['filter_atac_max_mito'] = metrics.atac_percent_mitochondrial <= THRESHOLD_ATAC_MAX_MITO
metrics['pass_all_filters'] = metrics.filter(like='filter_').all(axis=1)

# to collect all Thresholds here
print("THRESHOLD_ATAC_MIN_HQAA = {:,}".format(THRESHOLD_ATAC_MIN_HQAA))
print("THRESHOLD_ATAC_MIN_TSS_ENRICHMENT = {:,}".format(THRESHOLD_ATAC_MIN_TSS_ENRICHMENT))
print("THRESHOLD_ATAC_MAX_MITO = {:,}".format(THRESHOLD_ATAC_MAX_MITO))


##########
metrics = metrics.reset_index()
# List of pass-QC barcodes
pass_qc_nuclei = list(sorted(metrics[metrics.pass_all_filters].barcode.to_list()))


# Plot QC metrics #to work on plotting
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(3*4, 1*4))

ax = axs[0, 0]
atac_hqaa_vs_atac_tss_enrichment_plot(metrics, ax, alpha=0.02)
ax.axvline(THRESHOLD_ATAC_MIN_HQAA, color='red', ls='--', label='THRESHOLD_ATAC_MIN_HQAA = {:,}'.format(THRESHOLD_ATAC_MIN_HQAA))
ax.axhline(THRESHOLD_ATAC_MIN_TSS_ENRICHMENT, color='red', ls='--')
ax.legend()

ax = axs[0, 1]
barcode_rank_plot_atac(metrics, ax, alpha=0.02)
ax.axhline(THRESHOLD_ATAC_MIN_HQAA, color='red', ls='--')

ax = axs[0, 2]
atac_hqaa_vs_atac_mt_pct_plot(metrics, ax, alpha=0.02)
ax.axvline(THRESHOLD_ATAC_MIN_HQAA, color='red', ls='--')
ax.axhline(THRESHOLD_ATAC_MAX_MITO, color='green', ls='--', label='THRESHOLD_ATAC_MAX_MITO = {:,}'.format(THRESHOLD_ATAC_MAX_MITO))
ax.legend()

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

