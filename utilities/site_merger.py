import pandas as pd
import os
import shutil

# Combines sites that shared a weather station into one site each
# Sites were paired only where ta, vpd, ppfd_in and swc_shallow were numerically identical over their
# common record
# USA_HIL and CAN_TUR_P39 pair the same trees before and after a treatment date
# USA_SIL_OAK pairs the same trees across a sensor change, HD in 1PR against TSHB in 2PR
# Run after data_resampler.py, since the merge joins the sources on its half-hourly grid


resampled_directory = 'data/modeling_data/resampled'
archive_directory = 'data/modeling_data/resampled/merged_sources'

# Merged site name mapped to the sites that go into it
merge_groups = {
    'USA_HIL': ['USA_HIL_HF1_POS', 'USA_HIL_HF1_PRE', 'USA_HIL_HF2'],
    'CAN_TUR_P39': ['CAN_TUR_P39_POS', 'CAN_TUR_P39_PRE'],
    'USA_SIL_OAK': ['USA_SIL_OAK_1PR', 'USA_SIL_OAK_2PR'],
    'SWE_NOR_ST1_ST3': ['SWE_NOR_ST1_BEF', 'SWE_NOR_ST3'],
}

# Metadata file mapped to the column holding the site name, both are rebuilt by site_explorer.py
metadata_files = {
    'data/modeling_data/site_locations.csv': 'Site',
    'data/site_types.csv': 'Site',
}


def load_file(path):  # Returns the data indexed by timestamp and the interpolation flag on its own
    df = pd.read_csv(path, index_col='TIMESTAMP', parse_dates=['TIMESTAMP'])
    return df.drop(columns='interpolated'), df['interpolated']


def combine_flags(flags, index):  # A row is interpolated if it was interpolated in any of the sources
    aligned = [flag.reindex(index, fill_value=False) for flag in flags]  # reindexed before the concat so
    return pd.concat(aligned, axis=1).any(axis=1)  # the gaps come back as False rather than as NaN


def merge_features(sources):  # Returns one driver record for the location
    frames, flags = zip(*[load_file(f'{resampled_directory}/features/{site}_env_data.csv') for site in sources])
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.combine_first(frame)  # the four modelled drivers are identical wherever the
        # sources overlap, columns outside that set can differ a little and there the first source wins
    coverage = [frame.notna().any(axis=1).reindex(combined.index, fill_value=False) for frame in frames]
    combined['n_sources'] = pd.concat(coverage, axis=1).sum(axis=1)  # how many files covered each row,
    # the drivers agree so closely that nothing else in them records where they came from
    combined['interpolated'] = combine_flags(flags, combined.index)
    return combined


def merge_targets(sources):  # Returns every tree at the location in one frame
    frames, flags = zip(*[load_file(f'{resampled_directory}/targets/{site}_sapf_data.csv') for site in sources])
    combined = pd.concat(frames, axis=1)  # joined on timestamp, each column already named for its own site
    combined['interpolated'] = combine_flags(flags, combined.index)
    return combined


def merge_group(name, sources):  # Writes one merged site and archives the files that went into it
    features = merge_features(sources)
    targets = merge_targets(sources)
    features.to_csv(f'{resampled_directory}/features/{name}_env_data.csv')
    targets.to_csv(f'{resampled_directory}/targets/{name}_sapf_data.csv')
    for site in sources:  # moved rather than deleted, data_import reads every file in the directory
        shutil.move(f'{resampled_directory}/features/{site}_env_data.csv', f'{archive_directory}/{site}_env_data.csv')
        shutil.move(f'{resampled_directory}/targets/{site}_sapf_data.csv', f'{archive_directory}/{site}_sapf_data.csv')
    return len(features), len(targets.columns) - 1


def merge_metadata_rows(df, column, name, sources):  # Replaces the source rows with one row for the location
    rows = df[df[column].isin(sources)]
    if rows.empty:
        return df
    merged = rows.iloc[0].copy()
    averaged = [c for c in rows.select_dtypes('number').columns if c != 'Unnamed: 0']
    merged[averaged] = rows[averaged].mean()  # co-located, so these agree to several decimal places
    merged[column] = name
    if 'Type' in df.columns:  # one source can supply the functional type for the merged site
        named = [t for t in rows['Type'] if t != 'missing']
        merged['Type'] = named[0] if named else 'missing'
    df = df[~df[column].isin(sources)]
    return pd.concat([df, merged.to_frame().T], ignore_index=True)


def update_metadata():  # Puts the merged names into the files the clustering builds its site list from
    for path, column in metadata_files.items():
        df = pd.read_csv(path)
        for name, sources in merge_groups.items():
            df = merge_metadata_rows(df, column, name, sources)
        df = df.sort_values(column, ignore_index=True)
        df.to_csv(path, index=False)


def merge_all(verbose=True):
    os.makedirs(archive_directory, exist_ok=True)
    for name, sources in merge_groups.items():
        if os.path.exists(f'{resampled_directory}/features/{name}_env_data.csv'):
            continue  # this stage is meant to run once, so finished sites are skipped
        rows, sensors = merge_group(name, sources)
        if verbose:
            print(f'{name}: {len(sources)} sites into {rows} rows and {sensors} sap flux columns')
    update_metadata()


if __name__ == "__main__":
    merge_all()
