# Site selection

How the SAPFLUXNET sites used to train the models were chosen.

A **site** is one SAPFLUXNET record, with its own code, its own set of instrumented trees and its own data
file. A **location** is a group of sites within 1 km of each other, which therefore share a climate.
Records whose drivers were numerically identical were merged into a single site (step 4), so sites that
remain separate at a shared location differ in at least one driver, most often soil water content.
Location is the unit the training cap works on (see Training balance). The study has 60 sites at 43 locations.

| Step | Sites | Rows |
|---|---|---|
| Original study selection | 95 | |
| Sapwood-level sap flux available | 73 | |
| Excluded site list | 70 | |
| Co-located records merged | 65 | 1,770,140 |
| **Contributing to training** | **60** | **1,761,197** |

The final dataset is 60 sites at 43 distinct locations, 976 instrumented trees, and 1,761,197
half-hourly observations spanning 1995-04-20 to 2017-01-01.


## Selection steps

**1. Original selection (95 sites).** Drawn from SAPFLUXNET v0.1.5 (Poyatos et al. 2021, ESSD 13:2607)
by `data_explorer.py`, which kept sites carrying the required environmental drivers and recorded them
in `data/site_list.csv`. No sites have been added since this analysis was performed in 2022.

**2. Sapwood-level availability (95 → 73).** The study uses sapwood-level sap flux (cm h-1) so the
target is independent of tree size. 22 sites have no sapwood-level file in the database and were
dropped: 8 Czech, 3 ESP_ALT, 5 Italian, ISR_YAT_YAT, RUS_POG_VAR and the 4 USA_ORN plots. This removes the
Boreal forest biome entirely, since its only three sites (ITA_KAE_S20, ITA_MAT_S21 and ITA_RUN_N20) lacked sapwood data, and ISR_YAT_YAT, one of four Subtropical desert sites.

**3. Excluded site list (73 → 70).** SEN_SOU_POS, SEN_SOU_IRR and SEN_SOU_PRE have substantial sap flux
records but only 10 to 36 usable half hours of soil water content, leaving 7, 10 and 24 rows after the
join. After step 2 they were the only Subtropical desert sites left, so that biome is not represented in
the study.
Set in `excluded_sites` at the top of `site_explorer.py`.

**4. Co-location merges (70 → 65).** 38 of the 73 sapwood sites sit within 1 km of another site.
Records were merged only where `ta`, `vpd`, `ppfd_in` and `swc_shallow` were numerically identical over
their entire common record:

| Merged name | Source records |
|---|---|
| `USA_HIL` | HF1_POS, HF1_PRE, HF2 |
| `CAN_TUR_P39` | P39_POS, P39_PRE |
| `USA_SIL_OAK` | 1PR, 2PR |
| `SWE_NOR_ST1_ST3` | ST1_BEF, ST3 |

Nine records became four. No sap flux readings were lost: every tree column is retained and the site
average is taken over all trees present at each timestamp. Distinct locations are unchanged; the merge
removes double-counting of meteorology, not sites. Implemented in `site_merger.py`, with source files
archived to `resampled/merged_sources/`.

**5. Load-time criteria (65 listed → 60 contributing).** Applied in `data_import()` on every run, since
both depend on the active feature set.

*Minimum 30 days of usable observations, 1,440 half hours.* Excludes FRA_HES_HE1_NON (55 rows, 1.1
days) and FRA_HES_HE2_NON (43 rows, 0.9 days), both limited by soil water content rather than sap flux,
and ARG_MAZ (575 rows, 12.0 days) and ARG_TRE (623 rows, 13.0 days), whose records run for under two
weeks in November 2009.

*No more than 50% negative values.* SAPFLUXNET flags negative sap flux with `RANGE_WARN` but retains
the values, leaving treatment to the data user. A record
that is predominantly negative indicates a zero-flow baseline set too high rather than net transport.
Excludes USA_HUY_LIN_NON (58.6% negative, median -4.08 cm h-1 against a daytime median of 10.68). The
next highest proportion at any site is 3.7%.


## Composition

Five of the seven biomes in the original 95-site selection are represented. Boreal forest is lost in
step 2 and Subtropical desert in steps 2 and 3.

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
clusters. Location counts in the functional type table sum to 45 rather than 43 because two locations
host sites of more than one type: SWE_NOR (evergreen and mixed plots) and USA_TNO/USA_TNP (deciduous and
mixed). Group sizes remain uneven in both sites and locations; this is handled in the analysis design
rather than by further site removal.


## Training balance

Each cluster is capped so that no location contributes more than the 75th percentile of that cluster's
location record lengths (`cap_by_location` in `data_sanitizer.py`). A location above the cap is thinned
at random, with a fixed seed. A location at or below the cap it keeps every row. The cap is
computed separately for each cluster, so a site can contribute a different number of rows to its biome
cluster than to its functional type cluster.

| Biome | Cap | Rows before | Rows after |
|---|---|---|---|
| Temperate forest | 50,509 | 790,786 | 616,721 |
| Woodland/Shrubland | 51,648 | 608,294 | 430,267 |
| Temperate grassland desert | 244,582 | 244,582 | 244,582 |
| Tropical rain forest | 13,919 | 72,111 | 49,084 |
| Tropical forest savanna | 25,872 | 45,424 | 42,263 |
| **Total** | | **1,761,197** | **1,382,917** |

| Functional type | Cap | Rows before | Rows after |
|---|---|---|---|
| evergreen | 57,639 | 1,154,909 | 684,066 |
| mixed | 36,946 | 297,419 | 272,255 |
| deciduous | 21,132 | 294,950 | 231,220 |
| **Total** | | **1,747,278** | **1,187,541** |

The cap removes 21.5% of rows across the biome clusters and 32.0% across the functional type clusters.
Temperate grassland desert is a single location, so its cap equals its size and nothing is removed.

Within each capped cluster, 10% of rows are held out at random (seed 51) as the test set.
`test_set_builder.py` writes the held-out rows to `data/modeling_data/splits/`, so the random forest and
neural network are scored on the same rows.


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
