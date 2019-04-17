import pandas as pd
import os

def move_col_to_first(df, col_names):
    col_names.reverse()
    for col in col_names:
        df_col = df[col]
        df = df.drop(col,axis=1)
        df.insert(0,col,df_col)
    return df

def cnfname_extract(cnfpath):
    name = cnfpath.split('/')[-1]
    if 'features' in name:
        name =  name.split('_features')[0]
    if 'preprocessed' in name:
        name = name.split('_preprocessed')[0]
    if '.cnf' in name:
        return name
    else:
        return name+'.cnf'

file_csvs = []
datadir = 'newfeatures'
datatype = 'newfeatures'
# datadir = 'satzilla12features'
# datatype = 'satzilla12'
for file in os.listdir(datadir):
    file_csv = pd.read_csv(os.path.join(datadir,file))
    file_csvs.append(file_csv)
merged_df = pd.concat(file_csvs)
merged_df.drop(['Unnamed: 0'],axis=1,inplace = True)
if datatype=='newfeatures':
    merged_df['cnf_basename'] = merged_df['cnfname'].apply(cnfname_extract)
elif datatype=='satzilla12':
    merged_df['cnf_basename'] = merged_df['cnfname'].apply(cnfname_extract)
merged_df['train_test'] = merged_df['cnfname'].apply(lambda name: 'test'
                                                     if 'test' in name
                                                     else 'train')
merged_df['dataset'] = merged_df['cnfname'].apply(lambda name: name.split('_')[0])
merged_df = move_col_to_first(merged_df, ['cnfname','dataset','cnf_basename','train_test'])
merged_df.to_csv('newfeatures-all.csv',index=False)
# merged_df.to_csv('satzilla12features-all.csv',index=False)
