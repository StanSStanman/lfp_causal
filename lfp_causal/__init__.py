import os

if os.environ['USER'] == 'jerry':
    MCH = 'local'
elif os.environ['USER'] == 'rbasanisi':
    MCH = 'meso'

PRJ = 'lfp_causal'
