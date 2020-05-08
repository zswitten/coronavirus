import pandas as pd
import numpy as np
import ast

def make_dataset(inhibit_location, signature_location, gexp_location):

    inhibitiondf = pd.read_csv(inhibit_location)
    inhibitiondf['pert_id'] = inhibitiondf.pert_id.apply(lambda x: ast.literal_eval(x))
    inhibitiondf = inhibitiondf.explode('pert_id')

    signaturedf = pd.read_csv(signature_location, sep='\t')
    gexpdf = pd.read_csv(gexp_location)

    df = signaturedf.merge(inhibitiondf, on='pert_id')[['sig_id', 'pert_id', 'Inh_index']]
    df['gexp'] = df.sig_id.apply(lambda x: np.array(gexpdf[x]) if x in gexpdf else None)
    df = df.dropna(subset=['gexp'])

    x = np.array([z for z in df.gexp.values])
    y = np.array(df.Inh_index)
    return x, y

INHIBITION_TRAIN = 'data/inhibition_train.csv'
INHIBITION_VALID = 'data/inhibition_valid.csv'
INHIBITION_TEST = 'data/inhibition_test.csv'

SIGNATURE_PERT_ID_LOCATION = 'data/signature_perturbagen.txt'
GEXP_LOCATION = 'data/level5_1000.csv'

train_x, train_y = make_dataset(
    INHIBITION_TRAIN, SIGNATURE_PERT_ID_LOCATION, GEXP_LOCATION
)

valid_x, valid_y = make_dataset(
    INHIBITION_VALID, SIGNATURE_PERT_ID_LOCATION, GEXP_LOCATION
)

test_x, test_y = make_dataset(
    INHIBITION_TEST, SIGNATURE_PERT_ID_LOCATION, GEXP_LOCATION
)

