from lfp_causal.old.read_infos import files_info
from lfp_causal.old.raw import create_rawfiles, plot_rawdata
from lfp_causal.old.epochs import create_epochs
from lfp_causal.frequences_analysis import plot_psd

subject = ['freddie']
condition = ['easy']
# session = ['fneu0406', 'fneu0412', 'fneu0424', 'fneu0465', 'fneu0674', 'fneu0831', 'fneu0854', 'fneu0866', 'fneu0869',
#            'fneu0931', 'fneu0933', 'fneu0992', 'fneu1028', 'fneu1131', 'fneu1215', 'fneu1217', 'fneu1231', 'fneu1277',
#            'fneu1312', 'fneu1314']
# session = ['fneu0406']
session = None

def rawgen(subject, condition, session):
    for subj in subject:
        for cond in condition:
            if session == None:
                session = files_info(subj, cond)[0]
            for sess in session:
                # session = 'fneu'+session
                create_rawfiles(subj, cond, sess)

def epochsgen(subject, condition, session):
    for subj in subject:
        for cond in condition:
            if session == None:
                session = files_info(subj, cond)[0]
            for sess in session:
                # session = 'fneu'+session
                print('\n\n\n Processing session: {0} \n'.format(sess))
                create_epochs(subj, cond, sess)

def rawplot(subject, condition, session):
    for subj in subject:
        for cond in condition:
            if session == None:
                session = files_info(subj, cond)[0]
            for sess in session:
                plot_rawdata(subj, cond, sess)

def psd(subject, condition, session):
    for subj in subject:
        for cond in condition:
            if session == None:
                session = files_info(subj, cond)[0]
            for sess in session:
                plot_psd(subj, cond, sess)

if __name__ == '__main__':
    # create_folders(directory, subject, condition)
    # save_info(directory, xls_fname, subject, condition, info_dir, rawmat_dir)
    # rawgen(subject, condition, session)
    # epochsgen(subject, condition, session)
    # rawplot(subject, condition, session)
    psd(subject, condition, session)
