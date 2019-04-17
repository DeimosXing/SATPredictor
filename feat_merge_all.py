import pandas as pd
import os

newfeatures_dir = 'feature_res/newfeatures-all.csv'
sat12features_dir = 'feature_res/satzilla12features-all.csv'
runtimefeatures_dir = 'runtime_data/preprocessed_orig_runtimefeatures.csv'

newfeat = pd.read_csv(newfeatures_dir)
sat12feat = pd.read_csv(sat12features_dir)
rtfeat = pd.read_csv(runtimefeatures_dir)
# l = rtfeat[rtfeat['dataset']=='sc14-app']
# print(len(l))
# l2 = newfeat[newfeat['dataset']=='sc14-app']
# print(len(l2))
# l3 = sat12feat[sat12feat['dataset']=='sc14-app']
# print(len(l3))
temp = pd.merge(newfeat,sat12feat,on=['cnf_basename','dataset'])
# print(len(temp[temp['dataset']=='sc14-app']))
# temp.to_csv('temp.csv',index=False)
temp2 = pd.merge(temp,rtfeat,on=['cnf_basename','dataset'])
# temp2.to_csv('temp.csv',index=False)
# print(len(temp2))

orig_features_dir = 'feature_res/feature_comparison_after_permutation.csv'
origfeat = pd.read_csv(orig_features_dir)
origfeat = origfeat[origfeat['name'].str.contains('orig')==True]
origfeat['cnf_basename'] = origfeat['name'].apply(lambda name:name.split('/')[-1].split('_features')[0]+'.cnf')
temp3 = pd.merge(temp2, origfeat, on='cnf_basename',suffixes=('_preprocessed', '_orig'))
temp3.to_csv('data_all.csv',index=False)
pass

