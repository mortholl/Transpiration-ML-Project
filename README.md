# Transpiration ML Project

Machine learning models of transpiration built from SAPFLUXNET sap flux and environmental data,
fitted separately to clusters of sites grouped by climate, biome or plant functional type.

## Run order

All paths are relative to the repository root, so run every script from the root.

1. `utilities/data_explorer.py` — builds the site list from the available environmental variables
2. `utilities/site_explorer.py` — writes the site list with locations, species and functional types
3. `utilities/file_mover.py` — copies the matched environmental and sap flux files into `data/modeling_data`
4. `utilities/data_resampler.py` — puts every site onto a common half-hourly grid
5. `utilities/site_merger.py` — combines the sites that shared a weather station and rewrites the
   metadata to match
6. `utilities/test_set_builder.py` — draws the held out test rows, one split per cluster
7. a model script — `RandomForest/rf_optimization.py` or `Neural_Networks/ann_optimization.py`
8. `utilities/results_plotter.py` — plots the combined results written by the optimization scripts

`cluster_creator.py` builds the site clusters the models are grouped by. The model scripts import it, so
it does not need to be run on its own.

Steps 2 to 5 read the raw SAPFLUXNET v0.1.5 download, which is not kept in this repository; point
`source_directory` at the top of `site_explorer.py` and `file_mover.py` at its `csv/sapwood` folder.
Step 1 reads `data/plant`, an earlier layout of the download that is no longer available, so its output
in `data/site_list.csv` is committed and the step is not re-run. The outputs of steps 2 to 5 are
committed as well, so step 6 onward runs without the download.

Steps 4 and 5 skip work that is already finished, so both are safe to re-run. Step 2 rewrites the
metadata to one row per source site, so step 5 has to follow any re-run of it. Step 6 must be re-run
whenever the site data or the feature list changes; both model scripts read its splits, so they are
scored on the same rows.

`SITE_SELECTION.md` records which sites are in the study and why, and defines the distinction between a
site and a location.

## Layout

    data/modeling_data/features    environmental drivers, one file per site
    data/modeling_data/targets     sap flux, one column per tree, one file per site
    data/modeling_data/resampled   the same two folders on a half-hourly grid, written by step 5
    .../resampled/merged_sources   the per-site files that step 6 combined, kept out of the way
    utilities/                     data preparation, clustering and plotting
    RandomForest/ Neural_Networks/ SVM/ Penman_Monteith/    models
    archive/                       earlier versions, not in the run order
