import pandas as pd
import dataframe_image as dfi


ffffff = pd.read_csv(f'./jupyter_notebooks/XAI/GNNExplainer/feature_weights_final.csv', encoding="ISO-8859–1", dtype = str)
print(ffffff)


dfi.export(ffffff, './jupyter_notebooks/XAI/GNNExplainer/feature_weights_final.png')