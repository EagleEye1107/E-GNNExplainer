import os.path as osp

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from torch_geometric.data import HeteroData, download_url, extract_zip
from torch_geometric.transforms import RandomLinkSplit, ToUndirected

from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import StandardScaler
import category_encoders as ce



def load_node_csv(path, index_col, **kwargs):
    df = pd.read_csv(path, index_col = index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    return mapping

def load_edge_csv(df, src_index_col, src_mapping, dst_index_col, dst_mapping, **kwargs):
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_h = []
    for x in df["h"].values :
        edge_h.append(np.array(x))
    edge_h = np.array(edge_h)
    edge_h = torch.from_numpy(edge_h)
    
    return edge_index, edge_h




xai_datafile = "./input/Dataset/XAI/X_test3.csv"
df = pd.read_csv(xai_datafile, encoding="ISO-8859–1", dtype = str)
df['label'] = df['label'].apply(pd.to_numeric)
df["label"] = df["label"].apply(lambda x: int(x))
labels_column = df.label
# Preprocessing
encoder1 = ce.TargetEncoder(cols=[' Protocol',  'Fwd PSH Flags', ' Fwd URG Flags', ' Bwd PSH Flags', ' Bwd URG Flags'])
encoder1.fit(df, labels_column)
df = encoder1.transform(df)
scaler1 = StandardScaler()
cols_to_norm1 = list(set(list(df.iloc[:, :].columns )) - set(list(['label', ' Source IP', ' Destination IP'])) )
df[cols_to_norm1] = scaler1.fit_transform(df[cols_to_norm1])

# Just to create h
df['h'] = df[ cols_to_norm1 ].values.tolist()
df.drop(columns = cols_to_norm1, inplace = True)
columns_titles = [' Source IP', ' Destination IP', 'h', 'label']
df = df.reindex(columns=columns_titles)

print(df)


src_ip_mapping = load_node_csv(xai_datafile, index_col=' Source IP')
dst_ip_mapping = load_node_csv(xai_datafile, index_col=' Destination IP')

edge_index, edge_attr = load_edge_csv(
    df,
    src_index_col=' Source IP',
    src_mapping = src_ip_mapping,
    dst_index_col=' Destination IP',
    dst_mapping = dst_ip_mapping
)

print(edge_index)
print(edge_attr)

print()

print(ddddd)

data_d = Data(x = xai_datafile, edge_index = edge_index, edge_attr = edge_attr, y = df['label'])
print(data_d)