import pandas as pd
import numpy as np
import os

# Puts the driver and sap flux files onto a common half-hourly grid
# Site timesteps range from 5 minutes to an hour, so faster sites are averaged down and hourly sites
# are interpolated up; both fall out of resample() followed by a gap fill, with no branching per site
# Meant to run once, after file_mover.py and before any modeling


feature_directory = 'data/modeling_data/features'
target_directory = 'data/modeling_data/targets'
output_directory = 'data/modeling_data/resampled'

grid = '30min'
sum_columns = ['precip']  # accumulated over the interval, so summed rather than averaged
solar_columns = ['solar_TIMESTAMP', 'TIMESTAMP_solar']  # the column name is inconsistent between sites


def load_file(path):  # Returns a dataframe indexed by timestamp, ready to resample
    df = pd.read_csv(path)
    timestamps = pd.to_datetime(df['TIMESTAMP'], format='ISO8601', utc=True)
    df = df.drop(columns=[name for name in df.columns if name == 'TIMESTAMP' or name in solar_columns])
    df.index = timestamps.dt.round('min')  # clears the +/- 1 second jitter in the raw timestamps
    df.index.name = 'TIMESTAMP'
    return df.sort_index()  # some files interleave two series and are not in order


def fill_single_gaps(values):  # Returns the filled column and a flag marking which rows were filled
    # Fills gaps of one half hour and leaves anything longer alone.
    # limit= in interpolate() fills the first slot of a longer run, so run lengths are measured directly
    missing = values.isna()
    if missing.sum() == 0 or values.notna().sum() < 2:
        return values, pd.Series(False, index=values.index)
    run_lengths = missing.groupby((~missing).cumsum()).transform('sum')
    fillable = missing & (run_lengths == 1)
    filled = values.interpolate(method='pchip', limit_area='inside')  # shape preserving, will not overshoot
    filled[missing & ~fillable] = np.nan  # put back everything that was part of a longer gap
    return filled, fillable & filled.notna()


def resample_file(df):  # Returns the half-hourly dataframe and a flag marking interpolated rows
    aggregation = {name: ('sum' if name in sum_columns else 'mean') for name in df.columns}
    binned = df.resample(grid).agg(aggregation)
    interpolated = pd.Series(False, index=binned.index)
    for name in binned.columns:
        if name in sum_columns:  # splitting an accumulation over two slots invents a distribution
            continue
        binned[name], was_filled = fill_single_gaps(binned[name])
        interpolated = interpolated | was_filled  # gap structure differs by column, so check each one
    return binned, interpolated


def resample_site(site):  # Writes one site's resampled driver and sap flux files
    features = load_file(feature_directory + '/' + site + '_env_data.csv')
    targets = load_file(target_directory + '/' + site + '_sapf_data.csv')

    features, feature_flag = resample_file(features)
    targets, target_flag = resample_file(targets)
    features['interpolated'] = feature_flag  # kept separate so interpolated targets can be excluded
    targets['interpolated'] = target_flag     # from a test split later on

    features.to_csv(output_directory + '/features/' + site + '_env_data.csv')
    targets.to_csv(output_directory + '/targets/' + site + '_sapf_data.csv')
    return len(features), feature_flag.mean(), target_flag.mean()


def resample_all(force=False, verbose=True):
    os.makedirs(output_directory + '/features', exist_ok=True)
    os.makedirs(output_directory + '/targets', exist_ok=True)
    for filename in os.listdir(feature_directory):
        site = filename.split('_env')[0]
        if os.path.exists(output_directory + '/features/' + filename) and not force:
            continue  # this stage is meant to run once, so finished sites are skipped
        rows, feature_share, target_share = resample_site(site)
        if verbose:
            # a 60 minute site should come out near 50% interpolated and anything 30 minutes or
            # faster near 0%, so a number in between means the record has real gaps in it
            print(f'{site}: {rows} rows, {100 * feature_share:.0f}% drivers and '
                  f'{100 * target_share:.0f}% targets interpolated')


if __name__ == "__main__":
    resample_all()
