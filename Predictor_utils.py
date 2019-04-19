import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import *
from sklearn.ensemble import *
import numpy as np
import random
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

def heatmap(df):
    # corr heatmap
    corrmat = df.corr()
    f,ax = plt.subplots(figsize=(12,9))
    sns.heatmap(corrmat,vmax=.8, square=True)
    plt.show()

def feature_selection(model, X):
    print('shape before selection')
    print(X.shape)
    select_model = SelectFromModel(model, prefit=True)
    X_new = select_model.transform(X)
    print('shape after selection')
    print(X_new.shape)
    return X_new

def fit_val_reg(model, X, y):
    sfolder = KFold(n_splits=5, random_state=random.randint(1,100), shuffle=False)
    for train, test in sfolder.split(X, y):
        X_train,X_test,y_train,y_test = X.iloc[train],X.iloc[test],y.iloc[train],y.iloc[test]
        model.fit(X_train, y_train)
        y_pre = model.predict(X_test)
        print('r2 score:'+str(r2_score(y_test, y_pre))+' MSE:'+str(mean_squared_error(y_test,y_pre)))
        baseline = [y.mean()] * len(y_test)
        print('baseline score:' + str(r2_score(y_test, baseline))+' baseline MSE' + str(mean_squared_error(y_test,baseline)))

def fit_val_cls(model, X, y):
    sfolder = StratifiedKFold(n_splits=5, random_state=random.randint(1,100), shuffle=False)
    for train, test in sfolder.split(X, y):
        if isinstance(X,np.ndarray):
            X_train, X_test, y_train, y_test = X[train], X[test], y.iloc[train], y.iloc[test]
        else:
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
