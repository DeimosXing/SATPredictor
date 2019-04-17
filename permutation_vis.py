import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import stats
import statsmodels.api as sm
import re
import os

compare_3 = True

def plot_ecdf(input_data, label):
    ecdf_input = sm.distributions.ECDF(input_data)
    x_input = np.linspace(min(input_data), max(input_data))
    y_input = ecdf_input(x_input)
    plt.plot(x_input, y_input, label=label)

def get_results(input_dir):
    '''
    :param input_dir: directory of input file
    :return: a dict, key is the cnf file name, values are a list of numpy array, each array
    [input_branches, output_branches, input_syst, output_syst, int(input_realt), int(output_realt),isatis(optional),osatis(optional)]
    '''
    files = glob.glob(input_dir)
    files.sort()
    rst = {}
    for file in files:
        file_np = np.loadtxt(file, dtype=str)
        if len(file_np.shape) != 2:
            # result file contains just one row
            cnf_name = file_np[6].split('SATCompetition')[1]
            file_np = np.delete(file_np, 6)
            if cnf_name not in rst:
                rst[cnf_name] = []
            rst[cnf_name].append(file_np)
        else:
            # results file contains all data related with that cnf
            cnf_name = file_np[0,6].split('SATCompetition')[1]
            np.delete(file_np, 6)
            if cnf_name not in rst:
                rst[cnf_name] = list()
            for i in range(len(file_np)):
                rst[cnf_name].append(file_np[i])
    return rst

# data_dif_seed = get_results("../Cadical_noper_1-100/*")
data_preprocess_ins = get_results("./runtime_data/sc14crft_test/sc14crft_rndinit_maple/*")
data_dif_seed = get_results("./runtime_data/sc14crft_test/sc14crft_nopermute_maple/*")
data_permute_ins = get_results("./runtime_data/sc14crft_test/sc14crft_randper_maple/*")
# data_preprocess_ins = None
# store all data in numpy.array
dif_seed_np = None
preprocess_ins_np = None
permute_ins_np = None
# nbatch = 50
mean_var_rst = []
ks_rst = []
save_rst = True
rstfolder = 'rst_maple_rndinit'
if not os.path.exists('res/sc14crft_test/'+rstfolder):
    os.mkdir('res/sc14crft_test/'+rstfolder)
fix_shorter_rt=[]
fix_30percent_shorter_rt=[]
fix_extremeshort_rt=[]
runtime_ratio = []
fileindex = 0
for cnf_name, dif_seed_data in data_dif_seed.items():
    if cnf_name in data_permute_ins:
        permute_ins_data = data_permute_ins[cnf_name]
    else:
        continue
    if compare_3:
        if cnf_name in data_preprocess_ins:
            preprocess_ins_data = data_preprocess_ins[cnf_name]
        else:
            continue
    d_permute_ins = np.array(permute_ins_data)[:,3].astype(float)
    d_difseed = np.array(dif_seed_data)[:,3].astype(float)
    d_difseed = d_difseed[np.nonzero(d_difseed)]
    d_permute_ins = d_permute_ins[np.nonzero(d_permute_ins)]
    if not permute_ins_np is None:
        permute_ins_np = np.hstack((permute_ins_np,d_permute_ins))
        difseed_np = np.hstack((difseed_np,d_difseed))
    else:
        permute_ins_np = d_permute_ins
        difseed_np = d_difseed
    if compare_3:
        d_preprocess_ins = np.array(preprocess_ins_data)[:,3].astype(float)
        d_preprocess_ins = d_preprocess_ins[np.nonzero(d_preprocess_ins)]
        if len(d_preprocess_ins)==0:continue
        if not preprocess_ins_np is None:
            preprocess_ins_np = np.hstack((preprocess_ins_np,d_preprocess_ins))
        else:
            preprocess_ins_np = d_preprocess_ins
    if len(d_difseed)==0 or len(d_permute_ins)==0:
        continue
    title = cnf_name
    newtitle = ''
    newtitle = title[0:int(len(title)/3)] + '\n' + title[int(len(title)/3):int(len(title)/1.5)] +'\n'+title[int(len(title)/1.5):]
    # plot raw data
    plt.plot(d_permute_ins, label="permute_ins")
    plt.plot(d_difseed, label="difseed")
    if compare_3:
        plt.plot(d_preprocess_ins, label="preprocess_ins")
    # plt.axhline(ori_rt,label="original runtime")
    plt.title("{name}".format(i=fileindex, name=newtitle))
    plt.legend()
    plt.ylabel("runtime")
    if save_rst:plt.savefig("./res/sc14crft_test/{rst_folder}/rawdata_{num}".format(rst_folder=rstfolder,num=fileindex))
    # plt.show()
    plt.clf()

    # plot ecdf
    plot_ecdf(d_permute_ins, "permute_ins")
    plot_ecdf(d_difseed, "difseed")
    if compare_3:
        plot_ecdf(d_preprocess_ins, label="preprocess_ins")
    plt.title("{name}".format(name=newtitle))
    plt.legend()
    plt.title("ECDF {i}".format(i=fileindex))
    if save_rst:plt.savefig("./res/sc14crft_test/{rst_folder}/ecdf_{num}".format(rst_folder = rstfolder,num=fileindex))
    # plt.show()
    plt.clf()
    var_premute_ins = np.var(d_permute_ins)
    mean_permute_ins = np.mean(d_permute_ins)
    var_difseed = np.var(d_difseed)
    mean_difseed = np.mean(d_difseed)
    if compare_3:
        var_preprocess_ins = np.var(d_preprocess_ins)
        mean_preprocess_ins = np.mean(d_preprocess_ins)
        mean_var_rst.append([var_premute_ins, mean_permute_ins, var_difseed, mean_difseed, var_preprocess_ins, mean_preprocess_ins, permute_ins_data[0][2]])
    else:
        mean_var_rst.append([var_premute_ins,mean_permute_ins,var_difseed,mean_difseed,permute_ins_data[0][2]])
    runtime_ratio.append([float(fileindex), np.mean(d_permute_ins[np.nonzero(d_permute_ins)])/np.mean(d_difseed[np.nonzero(d_difseed)])])

    # save file names which has shorter runtime
    if mean_difseed<mean_permute_ins:
        fix_shorter_rt.append(title)
        if mean_difseed < 3:
            fix_extremeshort_rt.append(title)
    if mean_difseed<0.7* mean_permute_ins:
        fix_30percent_shorter_rt.append(title)

    # kstest
    sta,pvalue = stats.ks_2samp(d_permute_ins, d_difseed)
    ks_rst.append([sta,pvalue])

    fileindex += 1
# plot overall mean results
mv_rst = np.array(mean_var_rst).astype(float)
mv_rst = np.around(mv_rst, decimals=1)
plt.plot(sorted(mv_rst[:,1]), label ="mean_permute_ins")
plt.plot(sorted(mv_rst[:,3]), label ="mean_difseed")
if compare_3:
    plt.plot(sorted(mv_rst[:,5]), label="mean_preprocess_ins")
plt.plot(sorted(mv_rst[:,-1]), label ="original")
plt.legend()
plt.plot()
if save_rst:plt.savefig("./res/sc14crft_test/{rst_folder}/mean_rst".format(rst_folder=rstfolder))
plt.show()
plt.clf()
# plot overall variation results
plt.plot(sorted(mv_rst[:,0]), label ="var_permute_ins")
plt.plot(sorted(mv_rst[:,2]), label ="var_difseed")
if compare_3:
    plt.plot(sorted(mv_rst[:, 4]), label="var_preprocess_ins")
plt.legend()
plt.plot()
if save_rst:plt.savefig("./res/sc14crft_test/{rst_folder}/var_rst".format(rst_folder=rstfolder))
plt.show()
plt.clf()
# plot overall scatter plot
per_d =sorted(mv_rst[:,1])
dif_d =sorted(mv_rst[:,3])
if compare_3:
    pre_d=sorted(mv_rst[:,5])
plt.plot(per_d, dif_d, '.')
plt.yscale("log")
plt.xscale("log")
plt.xlabel("mean_permute_ins")
plt.ylabel("mean_difseed")
x_min = min(per_d+dif_d)
x_max = max(per_d+dif_d)
plt.xlim((x_min,x_max))
plt.ylim((x_min,x_max))
plt.plot(np.linspace(x_min, x_max),np.linspace(x_min, x_max))
plt.show()
# plt.savefig("./res/sc14crft_test/{rst_folder}/scatter_plot".format(rst_folder=rstfolder))
plt.clf()
# plot overall ecdf plot
plot_ecdf(difseed_np, "difseed")
plot_ecdf(permute_ins_np, "permute_ins")
if compare_3:
    plot_ecdf(preprocess_ins_np, "preprocess_ins")
plt.legend()
plt.title('overall ECDF')
plt.show()
# plt.savefig("./res/sc14crft_test/{rst_folder}/overall_ecdf".format(rst_folder=rstfolder))
plt.clf()
# percentage of fastest results for each class
mean_var_rst = np.array(mean_var_rst)
if compare_3:
    meandata_all = mean_var_rst[:,1:-1:2]
else:
    meandata_all = mean_var_rst[:,1:-1:2]
mean_argmax = np.argmax(meandata_all, axis=1)
print('permute_ins max ratio:'+str(sum(mean_argmax==0)/len(mean_argmax)))
print('difseed max ratio:'+str(sum(mean_argmax==1)/len(mean_argmax)))
print('preprocess_ins max ratio:'+str(sum(mean_argmax==2)/len(mean_argmax)))
# kstest
ks = np.array(ks_rst)
print(np.sum(ks[:,1]>0.05))
print(len(ks_rst))
print('ksrst:')
print(ks_rst)
plt.plot(ks[:,1], label="kstest")
plt.legend()
plt.show()
# np.savetxt("../dif_permu_seed_rst/firstrst/mean_var_rst.txt", mv_rst,header = "var_premute_ins/mean_fixseed/var_difseed/mean_difseed", fmt='%1.1f')
# np.savetxt("fnames_shorter_rt.txt",fix_shorter_rt,header="runtime longer after permutations", fmt='%s')
# np.savetxt("fnames_30%shorter_rt.txt",fix_30percent_shorter_rt,header="runtime longer after permutations(more than 10/7)", fmt='%s')
# np.savetxt("fnames_extreme_short.txt",fix_extremeshort_rt,header="runtime super short before permutations", fmt='%s')
# np.savetxt("runtime_ratio.txt", np.array(runtime_ratio),header='fixseed/difseed', fmt="%1.1f")
print(stats.ks_2samp(mv_rst[:,1], mv_rst[:,3]))
