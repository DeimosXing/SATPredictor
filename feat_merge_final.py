import pandas as pd
import os

features_dir = 'feature_res/satzilla12features-final.csv'
runtimefeatures_dir = 'runtime_data/preprocessed_orig_runtimefeatures.csv'

sat12feat = pd.read_csv(features_dir)
rtfeat = pd.read_csv(runtimefeatures_dir)
l = rtfeat[rtfeat['dataset']=='sc14-app']
print(len(l))
l3 = sat12feat[sat12feat['dataset']=='sc14-app']
print(len(l3))
temp2 = pd.merge(sat12feat,rtfeat,on=['cnf_basename'], how='left')
print(len(temp2))
# temp2.to_csv('data_final.csv',index=False)
pass

