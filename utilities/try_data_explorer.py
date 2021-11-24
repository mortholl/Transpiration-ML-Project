import pandas as pd

# This file examines the TRY data to find out which data we have for which species.
try_file = pd.read_csv('data/TRY_data.csv')
try_data = try_file[['TraitName', 'SpeciesName']]
species_traits = {}  # dictionary with keys of unique species, values of unique traits associated with the species
for i, row in try_data.iterrows():
    species = row['SpeciesName']
    trait = row['TraitName']
    if row['SpeciesName'] not in species_traits:
        species_traits.update({species: [trait]})
    if trait not in species_traits[species]:
        species_traits[species].append(trait)


df = pd.DataFrame.from_dict(species_traits, orient='index')
df.to_csv('data/try_explorer.csv')
