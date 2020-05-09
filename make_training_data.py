import pandas as pd
import numpy as np
import ast
import torch
import cmapPy.pandasGEXpress.parse_gctx as parse_gctx

def load_gctx(gctx_file):
    parsed = parse_gctx.parse(gctx_file)
    return parsed.data_df

def load_gexp(gexp_location):
    if gexp_location[-4:] == '.csv':
        return pd.read_csv(gexp_location)
    elif gexp_location[-5:] == '.gctx':
        return load_gctx(gexp_location)
    else:
        print("Improper gexp location")
        return None

def filter_inhibition_df(inhibitiondf):
    inhibitiondf['pert_id'] = inhibitiondf.pert_id.apply(lambda x: ast.literal_eval(x))
    inhibitiondf = inhibitiondf.explode('pert_id')
    return inhibitiondf.drop_duplicates(subset=['pert_id', 'Inh_index'])

def make_dataset(inhibit_location, signature_location, gexp_location):
    inhibitiondf = pd.read_csv(inhibit_location)
    inhibitiondf = filter_inhibition_df(inhibitiondf)

    signaturedf = pd.read_csv(signature_location, sep='\t')
    gexpdf = load_gexp(gexp_location)

    df = signaturedf.merge(inhibitiondf, on='pert_id')[['sig_id', 'pert_id', 'Inh_index']]
    df['gexp'] = df.sig_id.apply(lambda x: np.array(gexpdf[x]) if x in gexpdf else None)
    df = df.dropna(subset=['gexp'])

    x = torch.as_tensor(
        np.array([z for z in df.gexp.values]),
        dtype=torch.float32
    )
    y = torch.as_tensor(
        np.array(df.Inh_index),
        dtype=torch.float32
    )
    return x, y

INHIBITION_TRAIN = 'data/inhibition_train.csv'
INHIBITION_VALID = 'data/inhibition_valid.csv'
INHIBITION_TEST = 'data/inhibition_test.csv'

SIGNATURE_PERT_ID_LOCATION = 'data/signature_perturbagen.txt'
GEXP_LOCATION = 'data/level5_1000.csv'

def make_data():
    train_x, train_y = make_dataset(
        INHIBITION_TRAIN, SIGNATURE_PERT_ID_LOCATION, GEXP_LOCATION
    )

    valid_x, valid_y = make_dataset(
        INHIBITION_VALID, SIGNATURE_PERT_ID_LOCATION, GEXP_LOCATION
    )

    test_x, test_y = make_dataset(
        INHIBITION_TEST, SIGNATURE_PERT_ID_LOCATION, GEXP_LOCATION
    )
    return train_x, train_y, valid_x, valid_y, test_x, test_y

