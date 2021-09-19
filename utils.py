import numpy as np
import pandas as pd
import glob
import os
import re
import pickle
from tqdm import tqdm
from sklearn.metrics import r2_score
from numba_functions import *


# Data loaders

def load_trade_by_id(stock_id):
    parquet_path = glob.glob(f'./dataset/trade_train.parquet/stock_id={stock_id}/*')[0]
    df = pd.read_parquet(parquet_path)
    return df


def load_book_by_id(stock_id):
    parquet_path = glob.glob(f'./dataset/book_train.parquet/stock_id={stock_id}/*')[0]
    df = pd.read_parquet(parquet_path)
    return df


def load_train_by_id(stock_id):
    df = pd.read_csv(f'./dataset/train_/stock_id_{stock_id}.csv')
    return df


def get_path_by_id(type, stock_id):
    if type in ['book', 'trade']:
        return glob.glob(f'./dataset/{type}_train.parquet/stock_id={stock_id}/*')[0]
    else:
        print(f'Invalid type: {type}')
        return None


def load_trade():
    df_train = pd.read_csv('./dataset/train.csv')
    stock_ids = df_train.stock_id.unique().tolist()
    df_list = []
    for stock_id in tqdm(stock_ids):
        parquet_path = glob.glob(f'./dataset/trade_train.parquet/stock_id={stock_id}/*')[0]
        df = pd.read_parquet(parquet_path)
        df['stock_id'] = stock_id
        df_list.append(df)
    df_trade = pd.concat(df_list, ignore_index=True)
    return df_trade


def load_book():
    df_book = pd.read_csv('./dataset/train.csv')
    stock_ids = df_book.stock_id.unique().tolist()
    df_list = []
    for stock_id in tqdm(stock_ids):
        parquet_path = glob.glob(f'./dataset/book_train.parquet/stock_id={stock_id}/*')[0]
        df = pd.read_parquet(parquet_path)
        df['stock_id'] = stock_id
        df_list.append(df)
    df_book = pd.concat(df_list, ignore_index=True)
    return df_book


def load_both_by_id(stock_id, time_id):
    df_book = load_book_by_id(stock_id)
    df_trade = load_trade_by_id(stock_id)
    df_book = df_book.merge(df_trade, on=['time_id', 'seconds_in_bucket'], how='left')
    df_book = df_book.loc[df_book.time_id==time_id]
    df_book['wap'] = calc_wap_njit(
        df_book.bid_price1.values,
        df_book.ask_price1.values,
        df_book.bid_size1.values,
        df_book.ask_size1.values
    )
    df_book['log_return'] = calc_log_return(df_book['wap']).fillna(0)
    return df_book


def save_pickle(file, path):
    with open(path, 'wb') as f:
        pickle.dump(file, f)
    print('Done!')


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# Utility functions

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def gen_row_id(df):
    df['row_id'] = [f'{x[0]}-{x[1]}' for x in df[['stock_id', 'time_id']].values]
    return df


def get_fea_cols(df_train):
    return [f for f in df_train if re.match('[A-Z]+_', f)]


def save_result(df, filename):
    if 'row_id' not in df.columns:
        df = gen_row_id(df)
    df = df[['stock_id', 'time_id', 'row_id', 'target', 'pred']]
    df.to_csv(f'./results/{filename}.csv', index=False)


def save_models(list_model, pathname):
    for i, model in enumerate(list_model):
        model.save(f'./models/{pathname}/model_{i}.hdf5')


# Feature helpers

def calc_log_return(prices):
    return np.log(prices).diff()


def calc_realized_vol(log_returns):
    return np.sqrt(np.sum(np.power(log_returns, 2)))


# Metric calculators

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def calc_metric(df):
    r2_res = r2_score(y_true=df.target, y_pred=df.pred)
    rmspe_res = rmspe(y_true=df.target, y_pred=df.pred)
    print(f'   R2: {r2_res:.4f}')
    print(f'RMSPE: {rmspe_res:.4f}')


# Get ids

def get_time_id():
    list_time_id = sorted(load_book_by_id(0).time_id.unique())
    return list_time_id


def get_stock_id():
    list_stock_id = sorted([int(path.split('=')[1]) for path in glob.glob('./dataset/book_train.parquet/*')])
    return list_stock_id


# CV fold

def add_stratified_fold(df, n_fold):
    df_target = pd.read_csv('./dataset/train.csv')
    df_target = df_target.sort_values('target')
    df_target = gen_row_id(df_target)
    df_target['fold'] = np.arange(len(df_target)) % n_fold
    df = df.merge(df_target[['row_id', 'fold']], on='row_id')
    return df


def add_standard_fold(df, n_fold):
    df_target = pd.read_csv('./dataset/train.csv')
    df_target = gen_row_id(df_target)
    df_target['fold'] = np.arange(len(df_target)) % n_fold
    df = df.merge(df_target[['row_id', 'fold']], on='row_id')
    return df


def add_time_fold(df, n_fold, shuffle=False, seed=None):
    list_time_id = sorted(df['time_id'].unique())
    if shuffle:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(list_time_id)
    dict_time_id = dict(zip(list_time_id, range(len(list_time_id))))
    df['fold'] = df['time_id'].apply(lambda x: dict_time_id[x] % n_fold)
    return df


def add_funny_fold(df, n_fold):
    list_time_id = sorted(df['time_id'].unique())
    dict_time_id = dict(zip(list_time_id, range(len(list_time_id))))
    time_id_order = df['time_id'].apply(lambda x: dict_time_id[x] % n_fold)
    df['fold'] = (df['stock_id'] + time_id_order) % n_fold
    return df


def add_stock_fold(df, n_fold):
    df_target = pd.read_csv('./dataset/train.csv')
    list_stock_id = sorted(df_target['stock_id'].unique())
    dict_stock_id = dict(zip(list_stock_id, range(len(list_stock_id))))
    df_target = gen_row_id(df_target)
    df_target['fold'] = df_target['stock_id'].apply(lambda x: dict_stock_id[x] % n_fold)
    df = df.merge(df_target[['row_id', 'fold']], on='row_id')
    return df


def add_stock_time_fold(df, n_fold):
    df = add_stock_fold(df, n_fold)
    df = df.rename(columns={'fold': 'fold_s'})
    df = add_time_fold(df, n_fold)
    df = df.rename(columns={'fold': 'fold_t'})
    return df
