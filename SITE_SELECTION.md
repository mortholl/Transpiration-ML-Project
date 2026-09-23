# Site selection

How the SAPFLUXNET sites used to train the models were chosen.

| Step | Sites | Rows |
|---|---|---|
| Original study selection | 95 | |
| Sapwood-level sap flux available | 73 | |
| Excluded site list | 70 | |
| Co-located records merged | 65 | 1,770,042 |
| **Contributing to training** | **60** | **1,761,197** |

The final dataset is 60 sites at 43 distinct locations, 976 instrumented trees, and 1,761,197
half-hourly observations spanning 1995-04-20 to 2017-01-01.


## Selection steps

**1. Original selection (95 sites).** Drawn from SAPFLUXNET v0.1.5 (Poyatos et al. 2021, ESSD 13:2607)
by `data_explorer.py`, which kept sites carrying the required environmental drivers and recorded them
in `data/site_list.csv`. No sites have been added since.

**2. Sapwood-level availability (95 → 73).** The study uses sapwood-level sap flux (cm h-1) so the
target is independent of tree size. 22 sites have no sapwood-level file in the database and were
dropped: 8 Czech, 3 ESP_ALT, 6 Italian, ISR_YAT_YAT, RUS_POG_VAR and the 4 USA_ORN plots.

**3. Excluded site list (73 → 70).** SEN_SOU_POS, SEN_SOU_IRR and SEN_SOU_PRE have substantial sap flux
records but only 10 to 36 usable half hours of soil water content, leaving 7, 10 and 24 rows after the
join. They were the entire Subtropical desert biome, which is therefore not represented in the study.
Set in `excluded_sites` at the top of `site_explorer.py`.

**4. Co-location merges (70 → 65).** 38 of the 73 sapwood sites sit within 1 km of another site.
Records were merged only where `ta`, `vpd`, `ppfd_in` and `swc_shallow` were numerically identical over
their entire common record, indicating one weather station served both plots:

| Merged name | Source records |
|---|---|
| `USA_HIL` | HF1_POS, HF1_PRE, HF2 |
| `CAN_TUR_P39` | P39_POS, P39_PRE |
| `USA_SIL_OAK` | 1PR, 2PR |
| `SWE_NOR_ST1_ST3` | ST1_BEF, ST3 |

Nine records became four. No sap flux readings were lost — every tree column is retained and the site
average is taken over all trees present at each timestamp. Distinct locations are unchanged; the merge
removes double-counting of meteorology, not sites. Implemented in `site_merger.py`, with source files
archived to `resampled/merged_sources/`.

**5. Load-time criteria (65 listed → 60 contributing).** Applied in `data_import()` on every run, since
both depend on the active feature set.

*Minimum 30 days of usable observations, 1,440 half hours.* Excludes FRA_HES_HE1_NON (55 rows, 1.1
days) and FRA_HES_HE2_NON (43 rows, 0.9 days), both limited by soil water content rather than sap flux,
and ARG_MAZ (575 rows, 12.0 days) and ARG_TRE (623 rows, 13.0 days), whose records run for under two
weeks in November 2009. The next smallest contributing site is SWE_NOR_ST2 at 1,685 rows, 35.1 days, so
any threshold between 624 and 1,685 rows selects the same sites.

*No more than 50% negative values.* SAPFLUXNET flags negative sap flux with `RANGE_WARN` but retains
the values, leaving treatment to the data user. Small negative values are kept throughout; a record
that is predominantly negative indicates a zero-flow baseline set too high rather than net transport.
Excludes USA_HUY_LIN_NON (58.6% negative, median -4.08 cm h-1 against a daytime median of 10.68). The
next highest proportion at any site is 3.7%.


## Composition

| Biome | Sites | Locations | Rows |
|---|---|---|---|
| Temperate forest | 25 | 21 | 790,786 |
| Woodland/Shrubland | 22 | 14 | 608,294 |
| Temperate grassland desert | 3 | 1 | 244,582 |
| Tropical rain forest | 5 | 5 | 72,111 |
| Tropical forest savanna | 5 | 2 | 45,424 |

| Functional type | Sites | Locations | Rows |
|---|---|---|---|
| evergreen | 33 | 20 | 1,154,909 |
| mixed | 10 | 9 | 297,419 |
| deciduous | 16 | 15 | 294,950 |
| missing | 1 | 1 | 13,919 |

CRI_TAM_TOW has no functional type and is excluded from those clusters, but appears in the biome
clusters. Group sizes remain uneven in both sites and locations; this is handled in the training
design rather than by further site removal.


## Reproducing the site list

Run `site_explorer.py` → `file_mover.py` → `data_resampler.py` → `site_merger.py` from the repository
root. `site_explorer.py` writes the 70-site list, `site_merger.py` reduces it to 65, and the two
load-time criteria are re-evaluated on every training run. The list needs no manual editing.


## Data notes

- Sap flux is sapwood-level, cm3 cm-2 h-1 (cm h-1).
- All records are on a common half-hourly grid. Single half-hour gaps are filled by PCHIP interpolation
  and flagged; longer gaps are left missing. 9.5% of delivered driver rows and 8.6% of target rows
  contain an interpolated value.
- Sap flux above 200 cm h-1 is discarded as instrument error. This removed 7 readings, all at USA_HIL
  during a sensor malfunction on 2014-04-16; the next highest reading anywhere is 186.8 cm h-1.
