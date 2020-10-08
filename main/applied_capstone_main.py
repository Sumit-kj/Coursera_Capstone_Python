import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs


def get_dataframe(file_path):
    df = pd.read_csv(file_path)
    columns = ['SEVERITYCODE', 'ADDRTYPE', 'INTKEY', 'COLLISIONTYPE', 'PERSONCOUNT', 'PEDCOUNT', 'PEDCYLCOUNT',
               'VEHCOUNT', 'JUNCTIONTYPE', 'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 'ROADCOND', 'LIGHTCOND',
               'SPEEDING', 'HITPARKEDCAR']
    return df[columns], columns


def get_column_na_alt(col):
    fill_na_alt = {
        'SEVERITYCODE': 0,
        'ADDRTYPE': ' ',
        'INTKEY': 0,
        'COLLISIONTYPE': '',
        'PERSONCOUNT': 0,
        'PEDCOUNT': 0,
        'PEDCYLCOUNT': 0,
        'VEHCOUNT': 0,
        'JUNCTIONTYPE': '',
        'INATTENTIONIND': '',
        'UNDERINFL': '',
        'WEATHER': '',
        'ROADCOND': '',
        'LIGHTCOND': '',
        'SPEEDING': '',
        'HITPARKEDCAR': ''
    }
    return fill_na_alt[col]


def get_processed_dataframe(df, columns):
    column_map = {}
    print(df.head())
    categorical_columns = ['COLLISIONTYPE', 'JUNCTIONTYPE', 'WEATHER', 'ROADCOND', 'LIGHTCOND']
    one_hot_columns = ['ADDRTYPE', 'INATTENTIONIND', 'UNDERINFL', 'SPEEDING', 'HITPARKEDCAR']
    df['UNDERINFL'].replace('0', 'N', inplace=True)
    df['UNDERINFL'].replace('1', 'Y', inplace=True)
    df['INATTENTIONIND'].fillna('N', inplace=True)
    df['SPEEDING'].fillna('N', inplace=True)
    for col in categorical_columns:
        df.fillna(get_column_na_alt(col), inplace=True)
        column_map[col] = {}
        unique_values = df[col].unique()
        for i, uv in enumerate(unique_values):
            column_map[col][uv] = i
    df.replace(column_map, inplace=True)
    for col in one_hot_columns:
        df_temp = pd.get_dummies(df[col])
        df_temp_col = df_temp.columns
        df_col_rename = {}
        for col_r in df_temp_col:
            df_col_rename[col_r] = col + '_' + col_r.upper()
        df_temp.rename(columns= df_col_rename, inplace=True)
        df = df.join(df_temp)
        if '' in df_col_rename.keys():
            df.drop([col, str(col+'_').strip()], axis=1, inplace=True)
        else:
            df.drop([col], axis=1, inplace=True)
    return df, column_map


def get_column_correlation(df):
    correlation = df.corr()
    sbs.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns)
    plt.show()


def main():
    file_path = 'C:/Users/SumitKJ/PycharmProjects/AppliedDataScienceCapstone/data/Data-Collisions.csv'
    df_collision, columns = get_dataframe(file_path)
    df_collision, column_map = get_processed_dataframe(df_collision, columns)
    get_column_correlation(df_collision)


main()
