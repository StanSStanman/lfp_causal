from lfp_causal.old.read_infos import files_info
from lfp_causal.old.epochs import plot_epochs

subject = ['freddie']
condition = ['easy']
# session = ['fneu0406', 'fneu0428', 'fneu0429', 'fneu0430', 'fneu0626', 'fneu0469', 'fneu0762', 'fneu0854', 'fneu0491',
#            'fneu0495', 'fneu0751', 'fneu0963', 'fneu0968']#, 'fneu1131', 'fneu1215', 'fneu1217', 'fneu1231', 'fneu1277',
#            'fneu1312', 'fneu1314']
# session = ['fneu0508', 'fneu0616', 'fneu0873', 'fneu0945', 'fneu0978']
session = ['fneu0762']
# session = None
events_struct = {'trigger':([-0.5, 0.5], 'trigger'), 'cue':([-2.0, -1.0], 'trigger')}

if __name__ == '__main__':
    # create_folders(directory, subject, condition)
    # save_info(directory, xls_fname, subject, condition, info_dir, rawmat_dir)

    for subj in subject:
        for cond in condition:
            if session == None:
                session = files_info(subj, cond)[0]
    #
    #         # plot_area_evoked(subj, cond, session, 'limbic striatum', events_struct, picks=['mua'], mode='single')
    #         # plot_zscore_evoked(subj, cond, session, 'limbic striatum', events_struct, picks=['mua'], mode='max')
    #         # area_preprocessing(subj, cond, session, 'limbic striatum', pick=['mua'])
            for sess in session:
    #             # create_rawfiles(subj, cond, sess)
    #             create_epochs(subj, cond, sess)
    #             # plot_rawdata(subj, cond, sess)
    #             # plot_psd(subj, cond, sess)
    #             # plot_tfr(subj, cond, sess, 'trigger')
                    plot_epochs(subj, cond, sess, 'trigger', picks=['lfp'], all=True)
    #             # preprocessing(subj, cond, sess)



    # import os
    # for file in os.listdir('D:\\Databases\\db_lfp\\lfp_causal\\freddie\\easy\\raw_spike2'):
    #     sess = file.replace('_raw.fif', '')
    #     sess = 'fneu' + sess
    #     create_epochs('freddie', 'easy', sess)
