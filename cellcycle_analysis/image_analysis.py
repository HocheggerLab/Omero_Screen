# %% Import libraries

import pandas as pd
import numpy as np
from os import listdir
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import os
import math
from plotnine import *
import patchworklib as pw
from cell_cycle_distribution_functions import fun_normalise, fun_CellCycle

# %% Establishing path to the data and creating a folder to save exported .pdf files

path_data = "/Users/hh65/Desktop/221128_DepMap_Exp8_siRNAscreen_Plate1_72hrs/"
path_export = f"{path_data}Figures/"
os.makedirs(path_export, exist_ok=True)

# %% Establishing inconsistent wells which should be excluded from the analysis (optional)

wells_exclude = {}

# %% Importing RAW data & excluding inconsistent wells

list_files = list(filter(lambda file : ".csv" in file, listdir(path_data + "data/")))

data_raw = pd.DataFrame()

for file in list_files :
    
    tmp_data = pd.read_csv(path_data  + "data/" + file, sep = ",")
    tmp_plate = tmp_data["plate_id"].unique()[0]
    
    if tmp_plate in wells_exclude.keys() :

        tmp_data = tmp_data.copy().loc[~tmp_data["well_id"].isin(wells_exclude[tmp_plate])]
        
    data_raw = pd.concat([data_raw, tmp_data])
    del([tmp_data, file])
    
data_raw.loc[:, "cell_id"] = data_raw.groupby(["plate_id", "well_id", "image_id", "Cyto_ID"]).ngroup()

# %% Selecting parameters of interest and aggregating counts of nuclei and total cellular DAPI signal

data_IF = data_raw.groupby(["experiment", "plate_id", "well", "well_id", "image_id",
                            "cell_line", "condition", "Cyto_ID", "cell_id", "area_cell",
                            "intensity_mean_EdU_cell",
                            "intensity_mean_H3P_cell"]).agg(
                                         
                                         nuclei_count = ("label", "count"),
                                         nucleus_area = ("area_nucleus", "sum"),
                                         DAPI_total = ("integrated_int_DAPI", "sum")).reset_index()
                                
data_IF["condition"] = data_IF["condition"].astype(str)

# %% Normalising selected parameters & assigning cell cycle phases

data_IF = fun_normalise(data = data_IF, values = ["DAPI_total", "intensity_mean_EdU_cell", "intensity_mean_H3P_cell", "area_cell"])   
data_IF, data_thresholds = fun_CellCycle(data = data_IF, ctr_col = "condition", ctr_cond = "NT")

# %% Establishing proportions (%) of cell cycle phases

data_cell_cycle = pd.DataFrame()

for experiment in data_IF["experiment"].unique() :
    
    for cell_line in data_IF.loc[data_IF["experiment"] == experiment]["cell_line"].unique() :
    
        for condition in data_IF.loc[(data_IF["experiment"] == experiment) &
                                  (data_IF["cell_line"] == cell_line)]["condition"].unique() :
            
            tmp_data = data_IF.loc[(data_IF["experiment"] == experiment) &
                                   (data_IF["cell_line"] == cell_line) &
                                   (data_IF["condition"] == condition)]

            n = len(tmp_data)
            
            tmp_data = tmp_data.groupby(["experiment", "plate_id", "cell_line", "condition", "cell_cycle"],
                                        as_index = False).agg(
                                            count = ("cell_id", "count"),
                                            nuclear_area_mean = ("nucleus_area", "mean"),
                                            DAPI_total_mean = ("DAPI_total_norm", "mean"),
                                            area_cell_mean = ("area_cell_norm", "mean"))
            tmp_data["n"] = n
            tmp_data["percentage"] = (tmp_data["count"] / tmp_data["n"]) * 100
            data_cell_cycle = pd.concat([data_cell_cycle, tmp_data])
                    
data_cell_cycle_summary = data_cell_cycle.groupby(["cell_line", "cell_cycle", "condition"], as_index = False).agg(
    
    percentage_mean = ("percentage", "mean"),
    percentage_sd = ("percentage", "std"))

# %% Plotting & exporting combined EdU ~ DAPI scatter plots

for experiment in data_IF["experiment"].unique() :

    for cell_line in data_IF.loc[data_IF["experiment"] == experiment]["cell_line"].unique() :
    
        for condition in data_IF.loc[(data_IF["experiment"] == experiment) &
                                  (data_IF["cell_line"] == cell_line)]["condition"].unique() :
                 
            print(experiment + " – " + cell_line + " : " + condition)
            
            tmp_data = data_IF.loc[(data_IF["cell_line"] == cell_line) &
                                      (data_IF["condition"] == condition) &
                                      (data_IF["experiment"] == experiment)]
            
            tmp_thresholds = data_thresholds.loc[data_thresholds["cell_line"] == cell_line]
            
            sns.set_context(context = 'talk',
                            rc = {'font.size': 8.0,
                                  'axes.labelsize': 8.0,
                                  'axes.titlesize': 8.0,
                                  'xtick.labelsize': 8.0,
                                  'ytick.labelsize': 8.0,
                                  'legend.fontsize': 3,
                                  'axes.linewidth': 0.5,
                                  'grid.linewidth': 0.5,
                                  'lines.linewidth': 0.5,
                                  'lines.markersize': 2,
                                  'patch.linewidth': 0.5,
                                  'xtick.major.width': 0.5,
                                  'ytick.major.width': 0.5,
                                  'xtick.minor.width': 0.5,
                                  'ytick.minor.width': 0.5,
                                  'xtick.major.size': 5.0,
                                  'ytick.major.size': 5.0,
                                  'xtick.minor.size': 2.5,
                                  'ytick.minor.size': 2.5,
                                  'legend.title_fontsize': 3})
            
            Figure = sns.JointGrid(
                
                ratio = 3,
                xlim = (data_IF["DAPI_total_norm"].min(),
                        data_IF["DAPI_total_norm"].max()),
                ylim = (data_IF["intensity_mean_EdU_cell_norm"].min() - data_IF["intensity_mean_EdU_cell_norm"].min() * 0.2,
                        data_IF["intensity_mean_EdU_cell_norm"].max()))
            
            Figure.ax_joint.set_xscale("log")
            Figure.ax_joint.set_yscale("log")
            
            Figure.refline(y = tmp_thresholds["EdU_threshold"].values)
            Figure.refline(x = tmp_thresholds["DAPI_low_threshold"].values)
            Figure.refline(x = tmp_thresholds["DAPI_mid_threshold"].values)
            Figure.refline(x = tmp_thresholds["DAPI_high_threshold"].values)

            Figure.set_axis_labels("Integrated Hoechst intensity\n(normalised)\n", '\nMean EdU intensity\n(normalised)')
        
            sns.scatterplot(
                
                data = tmp_data,
                x = "DAPI_total_norm",
                y = "intensity_mean_EdU_cell_norm",
                color = '#000000',
                hue = "cell_cycle_detailed",
                palette = {"G1": "#6794db", "Early S": "#aed17d", "Late S": "#63a678", "G2": "#CC6677", "M": "#CC6677", "Polyploid" : "#b39bcf", "Polyploid (replicating)" : "#e3b344", "Sub-G1" : "#c7c7c7"},
                ec = "none",
                linewidth = 0,
                alpha = 0.1,
                legend = False,
                ax = Figure.ax_joint)
            
            sns.histplot(
                
                data = tmp_data,
                x = "DAPI_total_norm",
                color = "#ADACAC",
                ax = Figure.ax_marg_x,
                bins = 100,
                element = "step",
                stat = "density",
                fill = True)
            
            sns.histplot(
                
                data = tmp_data,
                y = "intensity_mean_EdU_cell_norm",
                color = "#ADACAC",
                ax = Figure.ax_marg_y,
                bins = 100,
                element = "step",
                stat = 'density',
                fill = True)
            
            Figure.ax_joint.text(
                data_IF["DAPI_total_norm"].min() + data_IF["DAPI_total_norm"].min() * 0.4,
                data_IF["intensity_mean_EdU_cell_norm"].max() - data_IF["intensity_mean_EdU_cell_norm"].max() * 0.5,
                f"{cell_line}\n{condition} µM",
                horizontalalignment = "left",
                size = 7,
                color = "#000000",
                weight ="normal")
            
            Figure.ax_joint.text(
                data_IF["DAPI_total_norm"].max() - data_IF["DAPI_total_norm"].min() * 0.4,
                data_IF["intensity_mean_EdU_cell_norm"].max() - data_IF["intensity_mean_EdU_cell_norm"].max() * 0.3,
                str(len(tmp_data)),
                horizontalalignment = "right",
                size = 7,
                color = "#000000",
                weight ="normal")
            
            Figure.fig.set_figwidth(2)
            Figure.fig.set_figheight(2)
            
            Figure.savefig(path_export + "EdU-DAPI_" + experiment + "_" + cell_line + "_" + condition + ".pdf", dpi = 300)
            Figure.savefig(path_export + "EdU-DAPI_" + experiment + "_" + cell_line + "_" + condition + ".png", dpi = 1000)
            del(tmp_data)
            
# %% Plotting & exporting distributions of H3-P signal in G2/M cells

for experiment in data_IF["experiment"].unique() :

    for cell_line in data_IF.loc[data_IF["experiment"] == experiment]["cell_line"].unique() :
    
        for condition in data_IF.loc[(data_IF["experiment"] == experiment) &
                                  (data_IF["cell_line"] == cell_line)]["condition"].unique() :
                     
            print(experiment + " – " + cell_line + " : " + condition)
            
            tmp_data = data_IF.loc[(data_IF["cell_line"] == cell_line) &
                                   (data_IF["condition"] == condition) &
                                   (data_IF["experiment"] == experiment) &
                                   (data_IF["cell_cycle_detailed"].isin(["G2", "M"]))]
            
            tmp_thresholds = data_thresholds.loc[data_thresholds["cell_line"] == cell_line]
            
            sns.set_context(context = 'talk',
                            rc = {'font.size': 8.0,
                                  'axes.labelsize': 8.0,
                                  'axes.titlesize': 8.0,
                                  'xtick.labelsize': 8.0,
                                  'ytick.labelsize': 8.0,
                                  'legend.fontsize': 3,
                                  'axes.linewidth': 0.5,
                                  'grid.linewidth': 0.5,
                                  'lines.linewidth': 0.5,
                                  'lines.markersize': 2.5,
                                  'patch.linewidth': 1.5,
                                  'xtick.major.width': 0,
                                  'ytick.major.width': 0.5,
                                  'xtick.minor.width': 0,
                                  'ytick.minor.width': 0.5,
                                  'xtick.major.size': 5.0,
                                  'ytick.major.size': 5.0,
                                  'xtick.minor.size': 2.5,
                                  'ytick.minor.size': 2.5,
                                  'legend.title_fontsize': 3})
            
            Figure = sns.JointGrid(
                
                ratio = 3,
                ylim = (data_IF["intensity_mean_H3P_cell_norm"].min() - data_IF["intensity_mean_H3P_cell_norm"].min() * 0.2,
                        data_IF["intensity_mean_EdU_cell_norm"].max()),)
            
            Figure.ax_joint.set_yscale("log")
            
            Figure.refline(y = tmp_thresholds["H3P_threshold"].values)

            Figure.set_axis_labels(" \n \n",
                                   "\nMean H3-P intensity\n(normalised)")
            
            sns.scatterplot(
                
                data = tmp_data,
                x = "cell_line",
                y = "intensity_mean_H3P_cell_norm",
                color = '#000000',
                hue = "cell_cycle_detailed",
                palette = {"G1": "#6794db", "Early S": "#aed17d", "Late S": "#63a678", "G2": "#CC6677", "M": "#fcba03", "Polyploid" : "#9230d9", "Debris" : "#a8a8a8"},
                ec = "none",
                linewidth = 0,
                alpha = 0.25,
                legend = False,
                ax = Figure.ax_joint)
            
            Figure.ax_joint.set_xticks([cell_line])
            Figure.ax_joint.set_xticklabels(["G2/M"])
            
            sns.histplot(
                
                data = tmp_data,
                y = "intensity_mean_H3P_cell_norm",
                color = "#ADACAC",
                ax = Figure.ax_marg_y,
                bins = 100,
                element = "step",
                stat = 'density',
                fill = True,
                lw = 0.5)
            
            Figure.fig.set_figwidth(0.5)
            Figure.fig.set_figheight(2)
            
            Figure.savefig(path_export + "H3P_" + experiment + "_" + cell_line + "_" + condition + ".pdf", dpi = 300)
            Figure.savefig(path_export + "H3P_" + experiment + "_" + cell_line + "_" + condition + ".png", dpi = 1000)
            del(tmp_data)
 
# %% Heatmaps of cell cycle phases

data_figure_cell_cycle = data_cell_cycle.copy()
data_figure_cell_cycle["cell_line"] = pd.Categorical(data_figure_cell_cycle["cell_line"], categories = ["U2OS", "RPE-1", "HeLa", "BJ1", "MM231"])
data_figure_cell_cycle["cell_cycle"] = pd.Categorical(data_figure_cell_cycle["cell_cycle"], categories = ["Sub-G1", "G1", "S", "G2/M", "Polyploid", "Polyploid (replicating)"])

Figure_heatmap_cell_cycle = (ggplot(data_figure_cell_cycle) +
                             
                             aes(x = "condition", y = "experiment", fill = "percentage") +
                             geom_tile(size = 0.4, colour = "#FFFFFF") +
                             facet_grid("cell_line ~ cell_cycle", space = "free_y",
                                        labeller = labeller(cols = {"Sub-G1" : "\nSub-G1", "G1" : "\nG1", "S" : "\nS", "G2/M" : "\nG2/M", "Polyploid" : "\nPolyploid", "Polyploid (replicating)" : "Polyploid\n(replicating)"})) +
                             
                             scale_fill_gradient2(low = "#a8d5e6", mid = "#fcba03", high = "#d41c34",
                                                 midpoint = 30,
                                                 name = "Proportion of all cells (%)",
                                                 breaks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) +
                             
                             labs(x = "Greatwall inhibitor (µM)",
                                  y = "Experiment") +
                             
                             guides(fill = guide_colourbar(barwidth = 3, barheight = 20, ticks = False)) +
                             
                             theme(
                                 subplots_adjust = {'wspace': 0.05, "hspace" : 0.05},
                                 panel_border = element_blank(),
                                 panel_background = element_rect(fill = "#FFFFFF"),
                                 panel_grid_major = element_blank(),
                                 panel_grid_minor = element_blank(),
                                 strip_background = element_blank(),
                                 strip_text = element_text(colour = "#000000", size = 8),
                                 axis_text = element_text(colour = "#000000", size = 8),
                                 axis_text_y = element_blank(),
                                 axis_text_x = element_text(angle = 90, vjust = 1),
                                 axis_title_x = element_text(colour = "#000000", size = 10),
                                 axis_title_y = element_text(colour = "#000000", size = 10),
                                 axis_ticks = element_blank(),
                                 
                                 legend_position = "top",
                                 legend_direction = "horizontal",
                                 legend_background = element_blank(),
                                 legend_title = element_text(colour = "#000000", size = 8),
                                 legend_title_align = ("center"),
                                 legend_key = element_blank(),
                                 legend_key_size = (8),
                                 legend_text = element_text(colour = "#000000", size = 8),
                                 legend_box_spacing = (0))
  
                             )

print(Figure_heatmap_cell_cycle)

ggsave(plot = Figure_heatmap_cell_cycle, filename = "Figure_heatmap_cell_cycle.pdf", path = path_export,
       width = 6, height = 2)

Figure_heatmap_DAPI_total = (ggplot(data_figure_cell_cycle) +
                             
                             aes(x = "condition", y = "experiment", fill = "DAPI_total_mean") +
                             geom_tile(size = 0.4, colour = "#FFFFFF") +
                             facet_grid("cell_line ~ cell_cycle", space = "free_y",
                                        labeller = labeller(cols = {"Sub-G1" : "\nSub-G1", "G1" : "\nG1", "S" : "\nS", "G2/M" : "\nG2/M", "Polyploid" : "\nPolyploid", "Polyploid (replicating)" : "Polyploid\n(replicating)"})) +
                             
                             scale_fill_gradient2(low = "#a8d5e6", mid = "#fcba03", high = "#d41c34",
                                                 midpoint = 4,
                                                 name = "Normalised integrated Hoechst intensity (mean)") +
                             
                             labs(x = "Greatwall inhibitor (µM)",
                                  y = "Experiment") +
                             
                             guides(fill = guide_colourbar(barwidth = 3, barheight = 20, ticks = False)) +
                             
                             theme(
                                 subplots_adjust = {'wspace': 0.05, "hspace" : 0.05},
                                 panel_border = element_blank(),
                                 panel_background = element_rect(fill = "#FFFFFF"),
                                 panel_grid_major = element_blank(),
                                 panel_grid_minor = element_blank(),
                                 strip_background = element_blank(),
                                 strip_text = element_text(colour = "#000000", size = 8),
                                 axis_text = element_text(colour = "#000000", size = 8),
                                 axis_text_y = element_blank(),
                                 axis_text_x = element_text(angle = 90, vjust = 1),
                                 axis_title_x = element_text(colour = "#000000", size = 10),
                                 axis_title_y = element_text(colour = "#000000", size = 10),
                                 axis_ticks = element_blank(),
                                 
                                 legend_position = "top",
                                 legend_direction = "horizontal",
                                 legend_background = element_blank(),
                                 legend_title = element_text(colour = "#000000", size = 8),
                                 legend_title_align = ("center"),
                                 legend_key = element_blank(),
                                 legend_key_size = (8),
                                 legend_text = element_text(colour = "#000000", size = 8),
                                 legend_box_spacing = (0))
  
                             )

print(Figure_heatmap_DAPI_total)

ggsave(plot = Figure_heatmap_DAPI_total, filename = "Figure_heatmap_DAPI_total.pdf", path = path_export,
       width = 6, height = 2)

Figure_heatmap_cell_area = (ggplot(data_figure_cell_cycle) +
                             
                             aes(x = "condition", y = "experiment", fill = "area_cell_mean") +
                             geom_tile(size = 0.4, colour = "#FFFFFF") +
                             facet_grid("cell_line ~ cell_cycle", space = "free_y",
                                        labeller = labeller(cols = {"Sub-G1" : "\nSub-G1", "G1" : "\nG1", "S" : "\nS", "G2/M" : "\nG2/M", "Polyploid" : "\nPolyploid", "Polyploid (replicating)" : "Polyploid\n(replicating)"})) +
                             
                             scale_fill_gradient2(low = "#a8d5e6", mid = "#fcba03", high = "#d41c34",
                                                 midpoint = 4.5,
                                                 name = "Normalised cell area (mean)") +
                             
                             labs(x = "Greatwall inhibitor (µM)",
                                  y = "Experiment") +
                             
                             guides(fill = guide_colourbar(barwidth = 3, barheight = 20, ticks = False)) +
                             
                             theme(
                                 subplots_adjust = {'wspace': 0.05, "hspace" : 0.05},
                                 panel_border = element_blank(),
                                 panel_background = element_rect(fill = "#FFFFFF"),
                                 panel_grid_major = element_blank(),
                                 panel_grid_minor = element_blank(),
                                 strip_background = element_blank(),
                                 strip_text = element_text(colour = "#000000", size = 8),
                                 axis_text = element_text(colour = "#000000", size = 8),
                                 axis_text_y = element_blank(),
                                 axis_text_x = element_text(angle = 90, vjust = 1),
                                 axis_title_x = element_text(colour = "#000000", size = 10),
                                 axis_title_y = element_text(colour = "#000000", size = 10),
                                 axis_ticks = element_blank(),
                                 
                                 legend_position = "top",
                                 legend_direction = "horizontal",
                                 legend_background = element_blank(),
                                 legend_title = element_text(colour = "#000000", size = 8),
                                 legend_title_align = ("center"),
                                 legend_key = element_blank(),
                                 legend_key_size = (8),
                                 legend_text = element_text(colour = "#000000", size = 8),
                                 legend_box_spacing = (0))
  
                             )

ggsave(plot = Figure_heatmap_cell_area, filename = "Figure_heatmap_cell_area.pdf", path = path_export,
       width = 6, height = 2)

print(Figure_heatmap_cell_area)

# %% Heatmaps of of polyploids

data_figure_cell_cycle = data_cell_cycle.copy()
data_figure_cell_cycle["cell_line"] = pd.Categorical(data_figure_cell_cycle["cell_line"], categories = ["U2OS", "RPE-1", "HeLa", "MM231"])
data_figure_cell_cycle["cell_cycle"] = pd.Categorical(data_figure_cell_cycle["cell_cycle"], categories = ["Sub-G1", "G1", "S", "G2/M", "Polyploid"])

Figure_heatmap_cell_cycle = (ggplot(data_figure_cell_cycle) +
                             
                             aes(x = "condition", y = "experiment", fill = "percentage") +
                             geom_tile(size = 0.4, colour = "#FFFFFF") +
                             facet_grid("cell_line ~ cell_cycle") +
                             
                             scale_fill_gradient2(low = "#a8d5e6", mid = "#fcba03", high = "#d41c34",
                                                 midpoint = 30,
                                                 name = "Proportion (%)",
                                                 breaks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) +
                             
                             labs(x = "Greatwall inhibitor (µM)",
                                  y = "Experiment") +
                             
                             guides(fill = guide_colourbar(barwidth = 3, barheight = 20, ticks = False)) +
                             
                             theme(
                                 subplots_adjust = {'wspace': 0.05, "hspace" : 0.05},
                                 panel_border = element_blank(),
                                 panel_background = element_rect(fill = "#FFFFFF"),
                                 panel_grid_major = element_blank(),
                                 panel_grid_minor = element_blank(),
                                 strip_background = element_blank(),
                                 strip_text = element_text(colour = "#000000", size = 8),
                                 axis_text = element_text(colour = "#000000", size = 8),
                                 axis_text_y = element_blank(),
                                 axis_text_x = element_text(angle = 90, vjust = 1),
                                 axis_title_x = element_text(colour = "#000000", size = 10),
                                 axis_title_y = element_text(colour = "#000000", size = 10),
                                 axis_ticks = element_blank(),
                                 
                                 legend_position = "top",
                                 legend_direction = "horizontal",
                                 legend_background = element_blank(),
                                 legend_title = element_text(colour = "#000000", size = 8),
                                 legend_title_align = ("center"),
                                 legend_key = element_blank(),
                                 legend_key_size = (8),
                                 legend_text = element_text(colour = "#000000", size = 8),
                                 legend_box_spacing = (0))
  
                             )

print(Figure_heatmap_cell_cycle)

ggsave(plot = Figure_heatmap_cell_cycle, filename = "Figure_heatmap_cell_cycle.pdf", path = path_export,
       width = 6, height = 1.8)













































    

























# %% Plotting proportions of replicating polyploid cells

Figure_polyploid_replicating = (ggplot(data_polyploid_summary) +
                                
                                aes(x = "condition", y = "replicating_perc_mean") +
                                geom_bar(position = "stack", stat = "identity", fill = "#b39bcf", colour = "#000000", size = 0.3, width = 0.8) +
                                geom_errorbar(aes(ymin = "replicating_perc_mean",
                                                  ymax = "replicating_perc_mean + replicating_perc_sd"),
                                              width = 0.5, position = position_dodge(.9), size = 0.3) +
                                
                                facet_wrap(" ~ cell_line",
                                           ncol = 1,
                                           labeller = labeller(cols = {"A" : "Sub-G1", "B" : "G1", "C" : "S", "D" : "G2/M", "E" : "Polyploid"}),
                                           scales = "free") +
                                
                                labs(x = " \nGreatwall inhibitor (µM)",
                                     y = "Proportion of replicating polyploid cells (%)\n ") +
                                
                                scale_y_continuous(expand = [0, 0], breaks = [0, 25, 50], limits = [0, 70]) +
                                scale_x_discrete(expand = [0.1, 0.1]) +
                                
                                scale_fill_manual(
                                    name = " ",
                                    breaks = ["A", "B", "C", "D", "E"],
                                    labels = ["Sub-G1", "G1", "S", "G2/M", "Polyploid"],
                                    values = ["#c7c7c7", "#6794db", "#aed17d", "#CC6677", "#b39bcf"]) +
                                
                                theme(
                                    subplots_adjust = {'wspace': 0.4, "hspace" : 0.9},
                                    panel_border = element_rect(colour = "#000000", size = 0.5),
                                    panel_background = element_rect(fill = "#FFFFFF"),
                                    panel_grid_major = element_blank(),
                                    panel_grid_minor = element_blank(),
                                    strip_background = element_blank(),
                                    strip_text = element_text(colour = "#000000", size = 8),
                                    axis_text = element_text(colour = "#000000", size = 8),
                                    axis_text_x = element_text(angle = 90),
                                    axis_title_x = element_text(colour = "#000000", size = 10),
                                    axis_title_y = element_text(colour = "#000000", size = 10),
                                    axis_ticks = element_line(colour = "#000000", size = 0.75),
                                    
                                    legend_position = "none",
                                    legend_direction = "horizontal",
                                    legend_background = element_blank(),
                                    legend_title = element_blank(),
                                    legend_title_align = ("center"),
                                    legend_key = element_blank(),
                                    legend_text = element_text(colour = "#000000", size = 12),
                                    legend_box_spacing = (0))
                                )

print(Figure_polyploid_replicating)

ggsave(plot = Figure_polyploid_replicating, filename = "Polyploid_replicating.pdf", path = path_export,
       width = 1, height = 4.7)