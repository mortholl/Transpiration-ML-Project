import pandas as pd
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import, cap_by_location
from utilities.test_set_builder import cluster_key, my_features

# Records how many rows each site and each location contributes to a cluster, before and after the cap
# Each cluster is loaded once with the cap disabled, then the cap is applied to the row labels on their
# own, which selects the same rows data_import would have kept
# Run on its own, nothing in the pipeline depends on its output

output_path = 'data/modeling_data/cap_analysis.csv'
cap_quantile = 0.75
seed = 51


def contributions(key, files):  # Returns one row per site and per location in the cluster
    x, y, info = data_import(my_features, files, verbose=False, cap_quantile=1.0, return_info=True)
    capped = cap_by_location(info, cap_quantile, seed, verbose=False)  # p100 above is a no op, so info
    cap = int(info.groupby('Location').size().quantile(cap_quantile))  # holds every usable row
    frames = []
    for level in ('Site', 'Location'):
        before = info.groupby(level).size()
        after = capped.groupby(level).size().reindex(before.index, fill_value=0)
        frames.append(pd.DataFrame({'Data set': key, 'Level': level, 'Name': before.index,
                                    'Before': before.values, 'After': after.values, 'Cap': cap}))
    return pd.concat(frames, ignore_index=True)


def analyse_all(verbose=True):
    cluster_creator = ClusterCreator.build_clusters()
    groups = zip(['func_', 'biome_'], [cluster_creator.func_cluster_dict, cluster_creator.biome_cluster_dict])
    frames = []
    for identifier, cluster_group in groups:
        for cluster in cluster_group:
            key = cluster_key(identifier, cluster)
            frame = contributions(key, cluster_group[cluster])
            frames.append(frame)
            if verbose:
                locations = frame[frame['Level'] == 'Location']
                print(f'{key}: cap {locations["Cap"].iloc[0]} rows, '
                      f'{locations["Before"].sum()} rows down to {locations["After"].sum()}, '
                      f'{(locations["After"] < locations["Before"]).sum()} of '
                      f'{len(locations)} locations thinned')
    pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
    if verbose:
        print(f'written to {output_path}')


if __name__ == "__main__":
    analyse_all()
