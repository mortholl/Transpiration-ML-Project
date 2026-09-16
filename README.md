# Transpiration ML Project

Machine learning models of transpiration built from SAPFLUXNET sap flux and environmental data,
fitted separately to clusters of sites grouped by climate, biome or plant functional type.

## Run order

All paths are relative to the repository root, so run every script from the root.

1. `utilities/data_explorer.py` — builds the site list from the available environmental variables
2. `utilities/site_explorer.py` — pulls site locations, species and functional types
3. `utilities/working_data_explorer.py` — pulls LAI where it is available
4. `utilities/file_mover.py` — copies the matched environmental and sap flux files into `data/modeling_data`
5. `utilities/data_resampler.py` — puts every site onto a common half-hourly grid, run once
6. `utilities/cluster_creator.py` — builds the site clusters the models are grouped by
7. a model script — `RandomForest/random_forest.py`, `Neural_Networks/ann.py` or `SVM/svm.py` for a single
   cluster, or `RandomForest/rf_optimization.py` / `Neural_Networks/ann_optimization.py` for every cluster
8. `utilities/results_plotter.py` — plots the combined results written by the optimization scripts

Steps 1 to 4 read from `data/plant` and `data/leaf`, which are the raw SAPFLUXNET download and are not
kept in this repository. Their outputs are committed, so steps 5 onward run without them.

`Penman_Monteith/pm_combined.py` generates the physical baseline and can be run any time after step 5.

## Layout

    data/modeling_data/features    environmental drivers, one file per site
    data/modeling_data/targets     sap flux, one column per tree, one file per site
    data/modeling_data/resampled   the same two folders on a half-hourly grid, written by step 5
    utilities/                     data preparation, clustering and plotting
    RandomForest/ Neural_Networks/ SVM/ Penman_Monteith/    models
    archive/                       earlier versions, not in the run order
