import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import numpy as np


X_test = pd.read_csv('./src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/Test.csv', encoding="ISO-8859–1", dtype = str)
X_test = X_test.apply(pd.to_numeric)
X_test = X_test.astype(float)

# print(X_test.dtypes.to_string())


# filename_expl = './src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/GNN_SHAP_explainer.sav'
filename = './src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/GNN_SHAP_shapvalues.sav'


# load_explainer = pickle.load(open(filename_expl, 'rb'))
# print(load_explainer)
load_shap_values = pickle.load(open(filename, 'rb'))
print(load_shap_values)

print(type(load_shap_values))

print("********")
print(len(load_shap_values))

print("+++++++++++")
print(len(load_shap_values.values))

print("+++++++++++")
print(len(load_shap_values.base_values))

print("+++++++++++")
print(len(load_shap_values.data[0]))

print("+++++++++++")
print(len(load_shap_values[0]))


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


def get_ABS_SHAP(df_shap, df):
    shap_v = df_shap
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
 
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    
    k2_f = k2[['Variable', 'SHAP_abs', 'Corr']]
    k2_f['SHAP_abs'] = k2_f['SHAP_abs'] * np.sign(k2_f['Corr'])
    k2_f.drop(columns='Corr', inplace=True)
    k2_f.rename(columns={'SHAP_abs': 'SHAP'}, inplace=True)

    return k2_f


foo_all = pd.DataFrame()

clss = [0, 1]

for k in clss:
    foo = get_ABS_SHAP(load_shap_values[k], X_test)
    foo['class'] = k
    foo_all = pd.concat([foo_all,foo])

import plotly_express as px
px.bar(foo_all,x='SHAP', y='Variable', color='class')