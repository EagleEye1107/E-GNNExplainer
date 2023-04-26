import xgboost
import shap
import matplotlib.pyplot as plt

# train an XGBoost model
X, y = shap.datasets.boston()
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
print(X)
print(y)
explainer = shap.Explainer(model)
shap_values = explainer(X)

shap.summary_plot(shap_values, X, show = False, max_display=X.shape[1])
plt.savefig('./notes/SHAP/graficcccc.png')
