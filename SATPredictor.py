import pandas as pd
import numpy as np
from lightgbm import *
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.svm import LinearSVC
import random
from Predictor_utils import *




data_dir = 'data_final_10000.csv'
data_df = pd.read_csv(data_dir)
data_df['runtime_diff'] = data_df['orig_runtime']-data_df['preprocessed_runtime']
data_df = data_df.select_dtypes(exclude=['object'])
# print(data_df.columns)

# pearson corr evaluation
# spearman(data_ratio_df, list(data_ratio_df.columns))


y_reg = data_df['runtime_diff']
data_df['runtime_diff_cls'] = data_df['runtime_diff'].apply(lambda diff:1 if diff >= 0 else 0)
data_df['missing'] = data_df['orig_runtime'].apply(lambda  runtime:1 if runtime==10000 else 0)
# y_cls = data_df['runtime_diff_cls']
y_cls = data_df['missing']
# drop columns that over 80% data are missing
data_df = data_df.replace([np.inf, -np.inf, -512], np.nan)
print('columns before dropping:'+str(len(data_df.columns)))
data_df = data_df.dropna(axis='columns', thresh=0.6*len(data_df))
print('columns after dropping:'+str(len(data_df.columns)))
# fill missing data with mean value
features = list(data_df.columns)
for feature in features:
    data_df[feature].fillna(data_df[feature].mean(), inplace=True)

standard_scale(data_df,data_df.columns.tolist())
# drop_list = ['runtime_diff','orig_runtime','preprocessed_runtime','runtime_diff_cls']
drop_list = ['orig_runtime','preprocessed_runtime','runtime_diff_cls','runtime_diff','missing']
data_df.drop(drop_list,axis=1,inplace=True)

# feature selection
selected_cols = ['nvarsOrig','nclausesOrig']
# selected_cols = ['nvars','nclauses','unit-featuretime','SP-unconstraint-q90','SP-unconstraint-q50','SP-unconstraint-mean','SP-unconstraint-max','sp-featuretime','SP-bias-q25','DIAMETER-entropy','DIAMETER-min','SP-bias-min']
data_df = data_df[selected_cols]

lsvc = LinearSVC(C=0.05, penalty='l1', dual=False).fit(data_df,y_cls)
# clf = ExtraTreesClassifier(n_estimators=100).fit(data_df,y_cls)
# print(clf.feature_importances_)
# data_df = feature_selection(lsvc, data_df)
X_train,X_test, y_train, y_test =train_test_split(data_df,y_cls,test_size=0.2, random_state=0)


# as a regression problem

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


# as a classification problem

# gbc = GradientBoostingClassifier()
rfc =RandomForestClassifier(n_estimators=10,
                            max_features=0.5,
                            min_samples_split=3)

lgbm_cls = LGBMClassifier(boosting_type='rf',learning_rate=0.1,n_estimators = 50,
                      bagging_freq=1,bagging_fraction=0.5,feature_fraction=0.5,
                      num_leaves=15,max_depth=6,min_data_in_leaf=5
             )
# X_train, X_test, y_train, y_test = train_test_split(data_df, y_cls, test_size=0.3)
fit_val_cls(lgbm_cls, X_train, y_train)

params_cls = {# 'max_depth':range(4,10,2),
              'num_leaves':range(5,20,2),
              'n_estimators':range(10,140,20),
              #'min_data_in_leaf':range(3,9,2)
}
gsearch = GridSearchCV(estimator=LGBMClassifier(boosting_type='rf',learning_rate=0.1,n_estimators = 10,
                      bagging_freq=1,bagging_fraction=0.5,feature_fraction=0.5,max_depth=6,min_data_in_leaf=5
             ), param_grid=params_cls,scoring='precision',cv=5)
# gsearch.fit(data_df, y_cls)
# print(gsearch.best_params_)
# print(gsearch.best_score_)

y_pred = lgbm_cls.predict(X_test)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print('Testset: accuracy:' + str(acc)+' precision' + str(precision)+' recall:' + str(recall)+' balanced accuracy:'+str(balanced_acc))











