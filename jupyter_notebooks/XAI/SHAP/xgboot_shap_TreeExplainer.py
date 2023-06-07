# Data Processing
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

# Modelling
import sklearn
from sklearn.model_selection import train_test_split
from scipy.stats import randint

import os


# XGBoost model
from xgboost import XGBClassifier
model = XGBClassifier()

path, dirs, files = next(os.walk("/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/input/Dataset/GlobalDataset/Splitted/"))
file_count = len(files)

for nb_files in range(1):
    data1 = pd.read_csv(f'{path}{files[nb_files]}', encoding="ISO-8859–1", dtype = str)
    print(f'{files[nb_files]} ++++++++++++++++++++++++++++++++++++++++++++++')

    # Delete two columns (U and V in the excel)
    cols = list(set(list(data1.columns )) - set(list(['Flow Bytes/s',' Flow Packets/s'])) )
    data1 = data1[cols]

    # IF We want to use IP Adr and do the mapping of the ip addresses because the random forest doesn't take in consideration
    # Mise en forme des noeuds
    data1[' Source IP'] = data1[' Source IP'].apply(str)
    data1[' Destination IP'] = data1[' Destination IP'].apply(str)

    # IP Mapping *************************************************************************
    # We do tha mapping of test set only because its faster and it will generate totally new nodes from the train set
    test_res = set()
    for x in list(data1[' Source IP']) :
        test_res.add(x)
    for x in list(data1[' Destination IP']) :
        test_res.add(x)

    test_re = {}
    cpt = 0
    for x in test_res:
        test_re[x] = cpt
        cpt +=1

    data1 = data1.replace({' Source IP': test_re})
    data1 = data1.replace({' Destination IP': test_re})

    print()
    # ***********************************************************************************

    # Delete unnecessary str data
    data1.drop(columns=['Flow ID',' Timestamp'], inplace=True)

    # data1 = data1.fillna(0)

    # -------------------- ????????????????????????????????????????? --------------------
    # simply do : nom = list(data1[' Label'].unique())
    nom = []
    nom = nom + [data1[' Label'].unique()[0]]
    for i in range(1, len(data1[' Label'].unique())):
        nom = nom + [data1[' Label'].unique()[i]]

    nom.insert(0, nom.pop(nom.index('BENIGN')))

    # Naming the two classes BENIGN {0} / Any Intrusion {1}
    data1[' Label'].replace(nom[0], 0,inplace = True)
    for i in range(1,len(data1[' Label'].unique())):
        data1[' Label'].replace(nom[i], 1,inplace = True)

    data1.rename(columns={" Label": "label"},inplace = True)
    label1 = data1.label
    data1.drop(columns=['label'],inplace = True)

    # ******** At this step data1 contains only the data without label column
    # ******** The label column is stored in the label variale 

    # Splitting the dataset to train and test sets
    X1_train, X1_test, y1_train, y1_test = train_test_split(data1, label1, test_size=0.3, random_state=123, stratify= label1)

    # for non numerical attributes (categorical data)
    # Since we have a binary classification, the category values willl be replaced with the posterior probability (p(target = Ti | category = Cj))
    # TargetEncoding is also called MeanEncoding, cuz it simply replace each value with (target_i_count_on_category_j) / (total_occurences_of_category_j)
    encoder1 = ce.TargetEncoder(cols=[' Protocol',  'Fwd PSH Flags', ' Fwd URG Flags', ' Bwd PSH Flags', ' Bwd URG Flags'])
    encoder1.fit(X1_train, y1_train)
    X1_train = encoder1.transform(X1_train)

    # scaler (normalization)
    scaler1 = StandardScaler()

    # Manipulate flow content (all columns except : label, Source IP & Destination IP)
    cols_to_norm1 = list(set(list(X1_train.iloc[:, :].columns )) - set(list(['label', ' Source IP', ' Destination IP'])) )
    X1_train[cols_to_norm1] = scaler1.fit_transform(X1_train[cols_to_norm1])

    # Model Training
    model.fit(X1_train, y1_train)
    print(model)

    # Model Testing
    X1_test = encoder1.transform(X1_test)
    X1_test[cols_to_norm1] = scaler1.transform(X1_test[cols_to_norm1])
    
    print("Model Testing")
    y_pred = model.predict(X1_test)

    print('Metrics : ')
    print("Accuracy : ", sklearn.metrics.accuracy_score(y1_test, y_pred))
    print("f1_score : ", sklearn.metrics.f1_score(y1_test, y_pred, labels = [0,1]))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")




label11 = y1_test.loc[label1 == 1].iloc[0:5000]
label10 = y1_test.loc[label1 == 0].iloc[0:5000]
X1_label = pd.concat([label11, label10], ignore_index = True)

X1_test1 = X1_test.loc[label11.index]
X1_test0 = X1_test.loc[label10.index]
X1_test = pd.concat([X1_test1, X1_test0], ignore_index = True)

X1_test2 =  pd.concat([X1_test, X1_label], axis=1)


# IP Mapping *************************************************************************
# We do tha mapping of test set only because its faster and it will generate totally new nodes from the train set
test_res = set()

for x in list(X1_test2[' Source IP']) :
    test_res.add(x)
for x in list(X1_test2[' Destination IP']) :
    test_res.add(x)

test_re = {}
cpt = 0.0
print("type(cpt)", type(cpt))
for x in test_res:
    test_re[x] = cpt
    cpt += 1.0

print("LAST type(cpt)", type(cpt))
print()

print(X1_test2)
X1_test2 = X1_test2.replace({' Source IP': test_re})
print("X1_test2 Source IP mapped")
X1_test2 = X1_test2.replace({' Destination IP': test_re})
print("X1_test2 Destination IP mapped")
print(X1_test2)
print()
# ***********************************************************************************

cols_to_norm1 = list(set(list(X1_test2.iloc[:, :].columns )) - set(list([' Source IP', ' Destination IP'])))

X1_test2[cols_to_norm1] = X1_test2[cols_to_norm1].apply(pd.to_numeric)
X1_test2[cols_to_norm1] = X1_test2[cols_to_norm1].astype(float)

print()
print(X1_test2.dtypes)

print()
print(X1_test2.dtypes.to_string())

print("----------")
print(len(X1_test2.columns))

X1_test2.to_csv(f'/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/Test_shap_xgb_Final.csv', sep=',', index = False)


# Prepare X_test by removing label
print(len(X1_test2.columns))
cols = list(set(list(X1_test2.columns )) - set(list(['label'])) )
X1_test2 = X1_test2[cols]
print(X1_test2.columns)
print(len(X1_test2.columns))


import shap

# XAI : TreeExplainer ######################
tree_explainer = shap.TreeExplainer(model)
tree_shap_values = tree_explainer.shap_values(X1_test2)


# Save shap values and explainer of TreeExplainer
import pickle
filename_expl = '/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/XGB_SHAP_explainer_TreeExplainer.sav'
pickle.dump(tree_explainer, open(filename_expl, 'wb'))
filename = '/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/XGB_SHAP_shapvalues_TreeExplainer.sav'
pickle.dump(tree_shap_values, open(filename, 'wb'))

print("TreeExplainer saved with pickle successfully")


# XAI : Explainer ######################
dflt_explainer = shap.Explainer(model)
dflt_shap_values = dflt_explainer(X1_test2)


# Save shap values and explainer of Explainer
import pickle
filename_expl = '/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/XGB_SHAP_explainer_explainer.sav'
pickle.dump(dflt_explainer, open(filename_expl, 'wb'))
filename = '/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/XGB_SHAP_shapvalues_explainer.sav'
pickle.dump(dflt_shap_values, open(filename, 'wb'))

print("Explainer saved with pickle successfully")



# XAI : Permutation Explainer ######################
cols_when_model_builds = model.get_booster().feature_names
X1_test2 = X1_test2[cols_when_model_builds]
perm_explainer = shap.Explainer(model.predict, X1_test2, algorithm = "permutation")
perm_shap_values = perm_explainer(X1_test2)


# Save shap values and explainer of Explainer
import pickle
filename_expl = '/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/XGB_SHAP_explainer_explainer_perm.sav'
pickle.dump(perm_explainer, open(filename_expl, 'wb'))
filename = '/home/ahmed/GNN-Based-ANIDS/GNN-Based-ANIDS/src/GNN_Model1/XP_CICIDS2017/XAI/SHAP_SAVED/XGB_SHAP_shapvalues_explainer_perm.sav'
pickle.dump(perm_shap_values, open(filename, 'wb'))

print("Perm Explainer saved with pickle successfully")
