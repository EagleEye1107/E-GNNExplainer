import xgboost
import shap
import matplotlib.pyplot as plt

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
