import xgboost
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# train an XGBoost model
X, y = shap.datasets.diabetes()
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
print(X)
print(y)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# shap.summary_plot(shap_values, X, show = False, max_display=X.shape[1])
# plt.savefig('./notes/SHAP/graficcccc.png')





# def get_ABS_SHAP(df_shap,df):
#     #import matplotlib as plt
#     # Make a copy of the input data
#     shap_v = pd.DataFrame(df_shap)
#     feature_list = df.columns
#     shap_v.columns = feature_list
#     df_v = df.copy().reset_index().drop('index',axis=1)
    
#     # Determine the correlation in order to plot with different colors
#     corr_list = list()
#     for i in feature_list:
#         b = np.corrcoef(shap_v[i],df_v[i])[1][0]
#         corr_list.append(b)
#     corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
 
#     # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
#     corr_df.columns  = ['Variable','Corr']
#     corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
#     shap_abs = np.abs(shap_v)
#     k=pd.DataFrame(shap_abs.mean()).reset_index()
#     k.columns = ['Variable','SHAP_abs']
#     k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
#     k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    
#     k2_f = k2[['Variable', 'SHAP_abs', 'Corr']]
#     k2_f['SHAP_abs'] = k2_f['SHAP_abs'] * np.sign(k2_f['Corr'])
#     k2_f.drop(columns='Corr', inplace=True)
#     k2_f.rename(columns={'SHAP_abs': 'SHAP'}, inplace=True)
    
#     return k2_f

# foo_all = pd.DataFrame()

# for k,v in list(enumerate(model.classes_)):

#     foo = get_ABS_SHAP(shap_values[k], X_test)
#     foo['class'] = v
#     foo_all = pd.concat([foo_all,foo])

# import plotly_express as px
# px.bar(foo_all,x='SHAP', y='Variable', color='class')








import pickle
filename_expl = './src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/SHAP_explainer.sav'
pickle.dump(explainer, open(filename_expl, 'wb'))
filename = './src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/SHAP_shapvalues.sav'
pickle.dump(shap_values, open(filename, 'wb'))



load_explainer = pickle.load(open(filename_expl, 'rb'))
print(load_explainer)
load_shap_values = pickle.load(open(filename, 'rb'))
print(load_shap_values)

shap.summary_plot(load_shap_values, X, show = False, max_display=X.shape[1])
plt.savefig('./notes/SHAP/graficccc_saved.png')
