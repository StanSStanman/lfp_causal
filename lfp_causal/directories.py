# In this script is possible to modify the folder in which the data are stored.
import os
s = os.sep

directory = 'D:'+s+'Databases'+s+'db_lfp'+s+'lfp_causal'+s
xls_fname = 'dataset_StriPAN_proba-{0}.xlsx'
rawmat_dir = directory + '{0}'+s+'{1}'+s+'raw_matlab'+s #{2}.mat' #.format(subject, condition, session)
info_dir = directory + '{0}'+s+'{1}'+s+'infos'+s #info_{2}.json' #.format(subject, condition, session)
neu_dir = directory + '{0}'+s+'{1}'+s+'neu_data'+s #{2}.npz' #.format(subject, condition, session)
beh_dir = directory + '{0}'+s+'{1}'+s+'beh_data'+s #{2}.npz' #.format(subject, condition, session)
# raw_dir = directory + '{0}'+s+'{1}'+s+'raw'+s
raw_dir = directory + '{0}'+s+'{1}'+s+'raw_spike2'+s
epochs_dir = directory + '{0}'+s+'{1}'+s+'epochs'+s+'{2}'+s
prep_dir = directory + '{0}'+s+'{1}'+s+'prep'+s
plot_dir = directory + '{0}'+s+'{1}'+s+'plots'+s