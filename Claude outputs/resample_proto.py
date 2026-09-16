"""Prototype: resample a SAPFLUXNET site pair onto a common half-hourly grid.

Not production code - a testbed to check the strategy holds on real sites.
"""
import pandas as pd, numpy as np

ROOT = '/mnt/user-data/uploads/Transpiration-ML-Project/data/modeling_data'
GRID = '30min'
SUM_COLS = {'precip'}
DROP_COLS = {'solar_TIMESTAMP', 'TIMESTAMP_solar'}


def load(path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df['TIMESTAMP'], format='ISO8601', utc=True).dt.round('min')
    df = df.drop(columns=[c for c in df.columns if c in DROP_COLS or c == 'TIMESTAMP'])
    df.index = ts
    df.index.name = 'TIMESTAMP'
    return df.sort_index()


def fill_single_slot_gaps(s, method='pchip'):
    """Interpolate only NaN runs of length 1; leave longer gaps alone."""
    isna = s.isna()
    if not isna.any() or s.notna().sum() < 2:
        return s, pd.Series(False, index=s.index)
    runlen = isna.groupby((~isna).cumsum()).transform('sum')
    fillable = isna & (runlen == 1)
    if not fillable.any():
        return s, pd.Series(False, index=s.index)
    try:
        out = s.interpolate(method=method, limit_area='inside')
    except Exception:
        out = s.interpolate(method='time', limit_area='inside')
    out[isna & ~fillable] = np.nan
    return out, fillable & out.notna()


def resample_site(df, method='pchip'):
    agg = {c: ('sum' if c in SUM_COLS else 'mean') for c in df.columns}
    binned = df.resample(GRID).agg(agg)
    observed = binned.notna()

    filled = binned.copy()
    interp_mask = pd.DataFrame(False, index=binned.index, columns=binned.columns)
    for c in binned.columns:
        if c in SUM_COLS:            # never interpolate an accumulation
            continue
        filled[c], interp_mask[c] = fill_single_slot_gaps(binned[c], method)
    return filled, observed, interp_mask


def report(site, feature_cols):
    env = load(f'{ROOT}/features/{site}_env_data.csv')
    sap_raw = load(f'{ROOT}/targets/{site}_sapf_data.csv')
    sap = pd.DataFrame({'sapf': sap_raw.mean(axis=1, skipna=True)})

    feats = [c for c in feature_cols if c in env.columns]
    env = env[feats]

    e_fill, e_obs, e_int = resample_site(env)
    s_fill, s_obs, s_int = resample_site(sap)

    joined = e_fill.join(s_fill, how='inner')
    joined['drivers_interpolated'] = e_int.reindex(joined.index).any(axis=1)
    joined['target_interpolated'] = s_int.reindex(joined.index).any(axis=1)
    usable_new = joined.dropna(subset=feats + ['sapf'])

    # what the current pipeline yields: positional join, no resample, dropna
    cur = env.copy()
    cur['sapf'] = sap['sapf'].values          # positional, as in data_sanitizer line 29
    usable_old = cur.dropna()

    step = pd.Series(e_fill.index).diff().dropna().dt.total_seconds().div(60)
    print(f'\n{"="*74}\n{site}   features={feats}\n{"="*74}')
    print(f'  native rows (env)          {len(env):>9,}')
    print(f'  half-hourly grid slots     {len(e_fill):>9,}   span {e_fill.index[0].date()} to {e_fill.index[-1].date()}')
    print(f'  grid uniform 30 min?       {bool((step == 30).all()):>9}   duplicates: {e_fill.index.duplicated().sum()}')
    print(f'  --- target column ---')
    print(f'    observed                 {int(s_obs["sapf"].sum()):>9,}  ({100*s_obs["sapf"].mean():5.1f}%)')
    print(f'    interpolated             {int(s_int["sapf"].sum()):>9,}  ({100*s_int["sapf"].mean():5.1f}%)')
    print(f'    left empty (gap > 1 hr)  {int(s_fill["sapf"].isna().sum()):>9,}  ({100*s_fill["sapf"].isna().mean():5.1f}%)')
    print(f'  --- usable rows after dropna ---')
    print(f'    current pipeline         {len(usable_old):>9,}')
    print(f'    resampled pipeline       {len(usable_new):>9,}   ({len(usable_new)/max(len(usable_old),1):.2f}x)')
    print(f'    of which target interp.  {int(usable_new["target_interpolated"].sum()):>9,}  ({100*usable_new["target_interpolated"].mean():5.1f}%)')
    print(f'    of which driver interp.  {int(usable_new["drivers_interpolated"].sum()):>9,}  ({100*usable_new["drivers_interpolated"].mean():5.1f}%)')
    return joined


if __name__ == '__main__':
    FEATS = ['ta', 'vpd', 'ppfd_in', 'swc_shallow']   # from ann.py
    for site in ['CHE_DAV_SEE', 'ZAF_SOU_SOU', 'USA_WIL_WC1', 'ESP_ALT_HUE']:
        report(site, FEATS)
