import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


cluster_info = pd.read_csv("data/modeling_data/cluster_info.csv")

# Scatter plots of cluster labels with climate statistics
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# x = cluster_info['MAP']
# y = cluster_info['MAT']
# z = cluster_info['Average Sap Flux']
# cmap = ListedColormap(sns.color_palette("husl"))
# color = cluster_info["K-Means Label"]
# sc = ax.scatter(x, y, z, cmap=cmap, c=color)
# ax.set_xlabel('MAP')
# ax.set_ylabel('MAT')
# ax.set_zlabel('Average Sap Flux')
# plt.legend(*sc.legend_elements(), loc=2)
# plt.show()
# plt.savefig("data/modeling_data/k_clusters.png")

# Box and whisker plot of R2 performance metric - RF
# rf_results = pd.read_csv("RandomForest/rf_results.csv", index_col=False)
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

# Box and whisker plot of R2 performance metric - NN
# ann_results = pd.read_csv("Neural_Networks/ann_results.csv", index_col=False)
# k_r2_box = []
# biome_r2_box = []
# func_r2_box = []
# for index, row in ann_results.iterrows():
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
# plt.title('Neural Network Test Set $R^2$')
# plt.savefig("Neural_Networks/results/ann_box_whisker.png")
# plt.clf()

# Box and whisker plot of feature importances - RF
# ta = rf_results[' ta'].tolist()
# vpd = rf_results['vpd'].tolist()
# ppfd_in = rf_results['ppfd_in'].tolist()
# swc_shallow = rf_results['swc_shallow'].tolist()
# fig = plt.figure()
# ax = fig.add_subplot()
# bp = ax.boxplot([ta, vpd, ppfd_in, swc_shallow])
# plt.ylabel('Relative Feature Importance')
# ax.set_xticklabels(['$T_a$', 'VPD', '$PPFD_{in}$', 'SWC'])
# plt.title('Random Forest Models: Average Feature Importances')
# plt.savefig('RandomForest/results/rf_feat_importances.png')

# Box and whisker plot of feature importances - NN
# ta = ann_results[' ta'].tolist()
# vpd = ann_results['vpd'].tolist()
# ppfd_in = ann_results['ppfd_in'].tolist()
# swc_shallow = ann_results['swc_shallow'].tolist()
# fig = plt.figure()
# ax = fig.add_subplot()
# bp = ax.boxplot([ta, vpd, ppfd_in, swc_shallow])
# plt.ylabel('Relative Feature Importance')
# ax.set_xticklabels(['$T_a$', 'VPD', '$PPFD_{in}$', 'SWC'])
# plt.title('Neural Network Models: Average Feature Importances')
# plt.savefig('Neural_Networks/results/ann_feat_importances.png')

# Pie charts of feature importances for every model
# rf_results = pd.read_csv("RandomForest/rf_results.csv", index_col=False)
# ann_results = pd.read_csv("Neural_Networks/ann_results.csv", index_col=False)
# labels = ['$T_a$', 'RH', 'VPD', 'PPFD', 'SWC']
#
# for i, model in enumerate(rf_results.iterrows()):
#     model = model[1]
#     name = model['Data set']
#     ta = model[' ta']
#     rh = model['rh']
#     vpd = model['vpd']
#     ppfd = model['ppfd_in']
#     swc = model['swc_shallow']
#     sizes = [ta, rh, vpd, ppfd, swc]
#     plt.pie(sizes, labels=labels, autopct='%1.1f%%', pctdistance=0.6)
#     plt.title(name)
#     plt.savefig('RandomForest/results/'+name+'_feat_importances')
#     plt.clf()
#
#
# for i, model in enumerate(ann_results.iterrows()):
#     model = model[1]
#     name = model['Data set']
#     ta = model[' ta']
#     rh = model['rh']
#     vpd = model['vpd']
#     ppfd = model['ppfd_in']
#     swc = model['swc_shallow']
#     sizes = [ta, rh, vpd, ppfd, swc]
#     plt.pie(sizes, labels=labels, autopct='%1.1f%%', pctdistance=0.6)
#     plt.title(name)
#     plt.savefig('Neural_Networks/results/'+name+'_feat_importances')
#     plt.clf()


