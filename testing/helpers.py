# -*- coding: utf-8 -*-
import pandas as pd


def preprocess_df_split_interval(df: pd.DataFrame) -> None:
    assert df.columns == ['faultsDurationTime']
    df['0'] = df.faultsDurationTime.apply(lambda x: str(x)[0])
    df['1'] = df.faultsDurationTime.apply(lambda x: str(x)[1:-1].split(',')[0])
    df['2'] = df.faultsDurationTime.apply(lambda x: str(x)[1:-1].split(',')[-1])
    df['3'] = df.faultsDurationTime.apply(lambda x: str(x)[-1])
    df['1'] = df['1'].astype(float)
    df['2'] = df['2'].astype(float)
    df.drop('faultsDurationTime', axis=1, inplace=True)

def preprocess_df_split_interval_and_factorize(df: pd.DataFrame) -> None:
    assert df.columns == ['faultsDurationTime']
    df['0'] = df.faultsDurationTime.apply(lambda x: str(x)[0])
    df['1'] = df.faultsDurationTime.apply(lambda x: str(x)[1:-1].split(',')[0])
    df['2'] = df.faultsDurationTime.apply(lambda x: str(x)[1:-1].split(',')[-1])
    df['3'] = df.faultsDurationTime.apply(lambda x: str(x)[-1])
    df['1'] = df['1'].astype(float)
    df['2'] = df['2'].astype(float)
    df['0'] = df['0'].map({'(': 1, '[': 2})
    df['3'] = df['3'].map({')': 3, ']': 4})
    df.drop('faultsDurationTime', axis=1, inplace=True)
