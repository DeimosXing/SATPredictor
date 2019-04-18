import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lightgbm import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.metrics import *
def spearman(df, cols):
    spr = pd.DataFrame()
    spr['feature'] = cols
    spr['corr'] = [df[f].corr(df['runtime_diff'],'spearman') for f in cols]
    spr = spr.sort_values('corr')
    plt.figure(figsize=(6,0.3*len(cols)))
    sns.barplot(data=spr, y='feature',x='corr',orient='h')
    plt.show()

def standard_scale(df, cols):
    std = StandardScaler()
    for col in cols:
        df[[col]] = std.fit_transform(df[[col]])

data_dir = 'data_final.csv'
data_df = pd.read_csv(data_dir)
data_df['runtime_diff'] = data_df['orig_runtime']-data_df['preprocessed_runtime']
# droplist = ['cnfname_x','dataset','cnf_basename','train_test_x','Unnamed: 0','name']
# droplist = ['cnfname_y','dataset_x','dataset_y','cnf_basename','train_test','cnfname_x']
# data_df.drop(droplist,axis=1,inplace=True)
# print(data_df.columns)
data_df = data_df.select_dtypes(exclude=['object'])
# print(data_df.columns)

# corr heatmap
# corrmat = data_df.corr()
# f,ax = plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat,vmax=.8, square=True)
# plt.show()



# pearson corr evaluation
# spearman(data_ratio_df, list(data_ratio_df.columns))


y_reg = data_df['runtime_diff']
data_df['runtime_diff_cls'] = data_df['runtime_diff'].apply(lambda diff:1 if diff >= 0 else 0)
y_cls = data_df['runtime_diff_cls']
# drop columns that over 80% data are missing
data_df = data_df.replace([np.inf, -np.inf, -512,0], np.nan)
print('columns before dropping:'+str(len(data_df.columns)))
data_df = data_df.dropna(axis='columns', thresh=0.6*len(data_df))
print('columns after dropping:'+str(len(data_df.columns)))
# fill missing data with mean value
features = list(data_df.columns)
for feature in features:
    data_df[feature].fillna(data_df[feature].mean(), inplace=True)

standard_scale(data_df,data_df.columns.tolist())
drop_list = ['runtime_diff','orig_runtime','preprocessed_runtime','runtime_diff_cls']
drop_list = ['orig_runtime','preprocessed_runtime','runtime_diff_cls']
data_df.drop(drop_list,axis=1,inplace=True)

# selected_cols = ['nvars','nclauses','unit-featuretime','SP-unconstraint-q90','SP-unconstraint-q50','SP-unconstraint-mean','SP-unconstraint-max','sp-featuretime','SP-bias-q25','DIAMETER-entropy','DIAMETER-min','SP-bias-min']
# data_df = data_df[selected_cols]


# as a regression problem
def fit_val_reg(model, X, y):
    sfolder = KFold(n_splits=5, random_state=0, shuffle=False)
    for train, test in sfolder.split(X, y):
        X_train,X_test,y_train,y_test = X.iloc[train],X.iloc[test],y.iloc[train],y.iloc[test]
        model.fit(X_train, y_train)
        y_pre = model.predict(X_test)
        print('r2 score:'+str(r2_score(y_test, y_pre))+' MSE:'+str(mean_squared_error(y_test,y_pre)))
        baseline = [y.mean()] * len(y_test)
        print('baseline score:' + str(r2_score(y_test, baseline))+' baseline MSE' + str(mean_squared_error(y_test,baseline)))

# gbr = GradientBoostingRegressor()
rfr = RandomForestRegressor(n_estimators=10,
                            max_features=0.5,
                            min_samples_split=3)
lgbm_reg = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=5,
                         learning_rate = 0.1, n_estimators=10 )
# fit_val_reg(lgbm_reg, data_df, y_reg)

params_reg = {'max_depth':range(6,14,2),
              'num_leaves':range(5,20,5),
              'n_estimators':range(40,200,20)}
gsearch = GridSearchCV(estimator=LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=5,
                         learning_rate = 0.1, n_estimators=10 ),
                       param_grid=params_reg,scoring='r2',cv=5)
# gsearch.fit(data_df, y_reg)
# print(gsearch.best_params_)
# print(gsearch.best_score_)
# {'max_depth': 6, 'n_estimators': 160, 'num_leaves': 15}
# 0.8896383029038976




# as a classification problem
def fit_val_cls(model, X, y):
    sfolder = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
    for train, test in sfolder.split(X, y):
        X_train,X_test,y_train,y_test = X.iloc[train],X.iloc[test],y.iloc[train],y.iloc[test]
        model.fit(X_train, y_train)
        y_pre = model.predict(X_test)
        acc = accuracy_score(y_test, y_pre)
        precision = precision_score(y_test, y_pre)
        recall = recall_score(y_test, y_pre)
        balanced_acc = balanced_accuracy_score(y_test, y_pre)
        print('accuracy:' + str(acc)+' precision' + str(precision)+' recall:' + str(recall)+' balanced accuracy:'+str(balanced_acc))
        baseline = [1]*len(y_test)
        acc = accuracy_score(y_test, baseline)
        print('fraction of True values:' + str(acc))

# gbc = GradientBoostingClassifier()
rfc =RandomForestClassifier(n_estimators=10,
                            max_features=0.5,
                            min_samples_split=3)

lgbm_cls = LGBMClassifier(boosting_type='rf',learning_rate=0.1,n_estimators = 100,
                      bagging_freq=1,bagging_fraction=0.5,feature_fraction=0.5,
                      num_leaves=5,max_depth=4,min_data_in_leaf=3
             )
fit_val_cls(lgbm_cls, data_df, y_cls)

params_cls = {'max_depth':range(4,14,2),
              'num_leaves':range(5,20,4),
              'n_estimators':range(20,200,20),
              'min_data_in_leaf':range(3,8,1)}
gsearch = GridSearchCV(estimator=LGBMClassifier(boosting_type='rf',learning_rate=0.1,n_estimators = 10,
                      bagging_freq=1,bagging_fraction=0.5,feature_fraction=0.5,
             ), param_grid=params_cls,scoring='balanced_accuracy',cv=5)
# gsearch.fit(data_df, y_cls)
# print(gsearch.best_params_)
# print(gsearch.best_score_)
# {'max_depth': 4, 'min_data_in_leaf': 3, 'n_estimators': 100, 'num_leaves': 5}
# 0.9985754985754985









