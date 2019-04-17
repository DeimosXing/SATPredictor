import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

data_dir = 'data_all.csv'
data_df = pd.read_csv(data_dir)
data_df['runtime_diff'] = data_df['orig_runtime']-data_df['preprocessed_runtime']
data_df.drop(['cnfname_x','dataset','cnf_basename','train_test_x','Unnamed: 0','name'],axis=1,inplace=True)

# corr heatmap
# corrmat = data_df.corr()
# f,ax = plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat,vmax=.8, square=True)
# plt.show()
def spearman(df, cols):
    spr = pd.DataFrame()
    spr['feature'] = cols
    spr['corr'] = [df[f].corr(df['runtime_diff'],'spearman') for f in cols]
    spr = spr.sort_values('corr')
    plt.figure(figsize=(6,0.3*len(cols)))
    sns.barplot(data=spr, y='feature',x='corr',orient='h')
    plt.show()

features = list(data_df.columns)
data_dif_df = pd.DataFrame()
data_ratio_df = pd.DataFrame()
for feature in features:
    if '_orig' in feature:
        prep_feat = feature.split('_orig')[0]+'_preprocessed'
        data_dif_df[feature.split('_orig')[0]+'_diff'] = data_df[feature]-data_df[prep_feat]
        data_ratio_df[feature.split('_orig')[0] + '_diff'] = data_df[feature] / data_df[prep_feat]

# pearson corr evaluation
# spearman(data_ratio_df, list(data_ratio_df.columns))

data_ratio_df = data_ratio_df.replace([np.inf, -np.inf], np.nan)
data_ratio_df = data_ratio_df.dropna(axis='columns', thresh=0.8*len(data_ratio_df))

# see how many columns were dropped
# print(len(data_ratio_df.columns))
# print(len(droped_df.columns))

for feature in data_ratio_df.columns.tolist():
    data_ratio_df[feature].fillna(data_ratio_df[feature].mean(), inplace=True)

# as a regression problem
X_train, X_test, y_train, y_test = train_test_split(data_ratio_df, data_df['runtime_diff'], test_size=0.3)
rfr = GradientBoostingRegressor()
rfr.fit(X_train, y_train)
y_predict = rfr.predict(X_test)
print(mean_squared_error(y_test,y_predict))

# as a classify problem
data_df['runtime_diff_cls'] = data_df['runtime_diff'].apply(lambda diff:1 if diff >= 0 else 0)
X_train, X_test, y_train, y_test = train_test_split(data_ratio_df, data_df['runtime_diff_cls'], test_size=0.3)
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_predict = gbc.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print(acc)








