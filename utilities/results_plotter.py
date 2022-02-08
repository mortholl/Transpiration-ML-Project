import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


cluster_info = pd.read_csv("data/modeling_data/cluster_info.csv")
# sns.scatterplot(x='MAP', y='MAT', data=cluster_info, hue='K-Means Label', palette="colorblind")
# plt.xlabel("Mean Annual Precipitation (mm)")
# plt.ylabel("Mean Annual Temperature (deg C)")
# plt.savefig("data/modeling_data/k_clusters.png")
# plt.cla()
# sns.scatterplot(x='MAP', y='MAT', data=cluster_info, hue='Biome', palette="colorblind")
# plt.xlabel("Mean Annual Precipitation (mm)")
# plt.ylabel("Mean Annual Temperature (deg C)")
# plt.savefig("data/modeling_data/biome_clusters.png")
# plt.clf()

rf_results = pd.read_csv("RandomForest/rf_results.csv")
# k_r2_box = []
# biome_r2_box = []
# func_r2_box = []
# for index, row in rf_results.iterrows():
#     if 'k_means' in row['Data set']:
#         k_r2_box.append(row[' R2 test'])
#     elif 'biome' in row['Data set']:
#         biome_r2_box.append(row[' R2 test'])
#     elif 'func' in row['Data set']:
#         func_r2_box.append(row[' R2 test'])
#
# fig = plt.figure()
# ax = fig.add_subplot()
# bp = ax.boxplot([k_r2_box, biome_r2_box, func_r2_box])
# plt.ylabel('$R^2$ of test set')
# ax.set_xticklabels(['K-Means',
#                     'Biome', 'Functional Type'])
# plt.title('Random Forest Test Set $R^2$')
# plt.savefig("RandomForest/results/rf_box_whisker.png")
# plt.clf()

ta = rf_results[' ta'].tolist()
rh = rf_results['rh'].tolist()
vpd = rf_results['vpd'].tolist()
ppfd_in = rf_results['ppfd_in'].tolist()
swc_shallow = rf_results['swc_shallow'].tolist()
precip = rf_results['precip'].tolist()
fig = plt.figure()
ax = fig.add_subplot()
bp = ax.boxplot([ta, rh, vpd, ppfd_in, swc_shallow, precip])
plt.ylabel('Relative Feature Importance')
ax.set_xticklabels(['$T_a$', 'RH', 'VPD', '$PPFD_{in}$', 'SWC', 'Precip'])
plt.title('Random Forest Models: Average Feature Importances')
plt.savefig('data/modeling_data/rf_feat_importances.png')
