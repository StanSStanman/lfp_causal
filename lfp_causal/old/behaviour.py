import pandas as pd

df = pd.read_csv('D:\\Databases\\db_lfp\\lfp_causal\\freddie\\easy\\behavior\\fneu1028.spk', delimiter='\t')
df.loc[df['0.2'].isin([300004, 300008, 300016])]