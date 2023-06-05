import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import numpy as np


X_test = pd.read_csv('./src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/Test_all.csv', encoding="ISO-8859–1", dtype = str)
X_test = X_test.apply(pd.to_numeric)
X_test = X_test.astype(float)

# print(X_test.dtypes.to_string())


# filename_expl = './src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/GNN_SHAP_explainer.sav'
filename = './src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/GNN_SHAP_shapvalues_all.sav'


# load_explainer = pickle.load(open(filename_expl, 'rb'))
# print(load_explainer)

label_column = X_test["label"]
attack_indx = []
benign_indx = []
for i, x in enumerate(label_column):
    if (x == 1.0):
        attack_indx.append(i)
    elif (x == 0.0):
        benign_indx.append(i)

load_shap_values = pickle.load(open(filename, 'rb'))

# shap.summary_plot(load_shap_values[attack_indx], feature_names = X_test.columns, show = False, max_display=X_test.shape[1])
# plt.savefig('./notes/SHAP/summary_attacks.png')
# plt.clf()
# shap.summary_plot(load_shap_values[benign_indx], feature_names = X_test.columns, show = False, max_display=X_test.shape[1])
# plt.savefig('./notes/SHAP/summary_benign.png')


# feature_order = np.abs(load_shap_values[attack_indx].values)

# feature_order = np.sum(np.abs(load_shap_values[attack_indx]), axis=0)
feature_order = np.argsort(np.sum(np.abs(load_shap_values[attack_indx].values), axis=0))
print([X_test.columns[i] for i in feature_order][::-1])

print(dddddddddddddddddddddddd)

# attacks scatters
shap.plots.scatter(load_shap_values[attack_indx," Bwd Packets/s"], show = False)
plt.savefig('./notes/SHAP/attacks_scatter/scatter1_attacks.png')
plt.clf()
shap.plots.scatter(load_shap_values[attack_indx,"Fwd Packets/s"], show = False)
plt.savefig('./notes/SHAP/attacks_scatter/scatter2_attacks.png')
plt.clf()
shap.plots.scatter(load_shap_values[attack_indx," Flow Duration"], show = False)
plt.savefig('./notes/SHAP/attacks_scatter/scatter3_attacks.png')
plt.clf()
shap.plots.scatter(load_shap_values[attack_indx," Flow IAT Mean"], show = False)
plt.savefig('./notes/SHAP/attacks_scatter/scatter4_attacks.png')
plt.clf()
shap.plots.scatter(load_shap_values[attack_indx," Flow IAT Max"], show = False)
plt.savefig('./notes/SHAP/attacks_scatter/scatter5_attacks.png')
plt.clf()
shap.plots.scatter(load_shap_values[attack_indx," Flow IAT Min"], show = False)
plt.savefig('./notes/SHAP/attacks_scatter/scatter6_attacks.png')
plt.clf()
shap.plots.scatter(load_shap_values[attack_indx," Packet Length Mean"], show = False)
plt.savefig('./notes/SHAP/attacks_scatter/scatter7_attacks.png')
plt.clf()
shap.plots.scatter(load_shap_values[attack_indx," Packet Length Variance"], show = False)
plt.savefig('./notes/SHAP/attacks_scatter/scatter8_attacks.png')
plt.clf()
shap.plots.scatter(load_shap_values[attack_indx," Average Packet Size"], show = False)
plt.savefig('./notes/SHAP/attacks_scatter/scatter9_attacks.png')
plt.clf()
shap.plots.scatter(load_shap_values[attack_indx,"Total Length of Fwd Packets"], show = False)
plt.savefig('./notes/SHAP/attacks_scatter/scatter10_attacks.png')
plt.clf()

# benign scatters
shap.plots.scatter(load_shap_values[benign_indx," Flow IAT Max"], show = False)
plt.savefig('./notes/SHAP/benign_scatter/scatter1_benign.png')
plt.clf()
shap.plots.scatter(load_shap_values[benign_indx,"Fwd Packets/s"], show = False)
plt.savefig('./notes/SHAP/benign_scatter/scatter2_benign.png')
plt.clf()
shap.plots.scatter(load_shap_values[benign_indx," Flow Duration"], show = False)
plt.savefig('./notes/SHAP/benign_scatter/scatter3_benign.png')
plt.clf()
shap.plots.scatter(load_shap_values[benign_indx," Flow IAT Mean"], show = False)
plt.savefig('./notes/SHAP/benign_scatter/scatter4_benign.png')
plt.clf()
shap.plots.scatter(load_shap_values[benign_indx," Bwd Packets/s"], show = False)
plt.savefig('./notes/SHAP/benign_scatter/scatter5_benign.png')
plt.clf()
shap.plots.scatter(load_shap_values[benign_indx," Average Packet Size"], show = False)
plt.savefig('./notes/SHAP/benign_scatter/scatter6_benign.png')
plt.clf()
shap.plots.scatter(load_shap_values[benign_indx," Packet Length Mean"], show = False)
plt.savefig('./notes/SHAP/benign_scatter/scatter7_benign.png')
plt.clf()
shap.plots.scatter(load_shap_values[benign_indx," Avg Fwd Segment Size"], show = False)
plt.savefig('./notes/SHAP/benign_scatter/scatter8_benign.png')
plt.clf()
shap.plots.scatter(load_shap_values[benign_indx," Subflow Fwd Bytes"], show = False)
plt.savefig('./notes/SHAP/benign_scatter/scatter9_benign.png')
plt.clf()
shap.plots.scatter(load_shap_values[benign_indx," Fwd Packet Length Mean"], show = False)
plt.savefig('./notes/SHAP/benign_scatter/scatter10_benign.png')
plt.clf()

print(rrrrrrrrrrrrrrr)

# shap_attack = pd.DataFrame(columns = ["Edge_ID", "Shap_Value"])
shap_attack = []
for indx in attack_indx:
    shap_attack.append((indx, load_shap_values[indx]))  # adding a row

# shap_benign = pd.DataFrame(columns = ["Edge_ID", "Shap_Value"])
shap_benign = []
for indx in attack_indx:
    shap_benign.append((indx, load_shap_values[indx]))  # adding a row


print("********************")
print(shap_attack[0])
print("+++++++++++++++")
print(shap_benign[0])
print(dddddddddddddddddddddddddddddddddddd)


# shap.summary_plot(load_shap_values, X_test, plot_type="bar")

shap.summary_plot(load_shap_values[1], X_test.values, feature_names = X_test.columns)
shap.summary_plot(load_shap_values[0], X_test.values, feature_names = X_test.columns)

# shap.plots.bar(load_shap_values[:,:, 0])

# print(len(load_shap_values.values))
# print(len(load_shap_values.values[0]))
# print(len(load_shap_values.base_values))
# print(len(load_shap_values.data))

# shap.plots.waterfall(load_shap_values[0], max_display=X_test.shape[1]+10)
# shap.summary_plot(load_shap_values, X_test, show = False, max_display=X_test.shape[1])

# shap.plots.waterfall(load_shap_values[0], show = False, max_display=X_test.shape[1])

# shap.plots.scatter(load_shap_values[:," Flow IAT Std"], color = load_shap_values)

shap.plots.scatter(load_shap_values[:," Flow IAT Std"])

# plt.savefig('./notes/SHAP/scatter.png')

# shap.plots.heatmap(load_shap_values)

# visualize the first prediction's explanation with a force plot
# shap.plots.force(load_shap_values.base_values[0])

# visualize all the training set predictions
# shap.plots.force(load_shap_values.base_values, load_shap_values)