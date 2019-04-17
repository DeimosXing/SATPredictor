import pandas as pd
import os


def move_col_to_first(df, col_names):
    col_names.reverse()
    for col in col_names:
        df_col = df[col]
        df = df.drop(col,axis=1)
        df.insert(0,col,df_col)
    return df

def dataset(cnfname):
    if '2018_SAT_Competition' in cnfname:
        return 'sc18'
    elif '2017_SAT_Competition' in cnfname:
        return 'sc17'
    else:
        return cnfname.split('/')[-2]
runtime_df = pd.read_csv('preprocessing_runtimes.csv')
cnfname = runtime_df['instance']
runtime_df['cnfname']=cnfname
runtime_df.drop(['instance'],axis=1,inplace = True)
runtime_df['cnf_basename'] = runtime_df['cnfname'].apply(lambda name: name.split('/')[-1])
runtime_df['dataset'] = runtime_df['cnfname'].apply(dataset)
runtime_df = move_col_to_first(runtime_df, ['cnfname','dataset','cnf_basename'])
runtime_df.rename(columns={'orig':'orig_runtime','preprocessed':'preprocessed_runtime'}, inplace=True)
runtime_df.to_csv('preprocessed_orig_runtimefeatures.csv',index=False)
