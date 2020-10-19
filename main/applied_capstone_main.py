import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_dataframe(file_path):
    df = pd.read_csv(file_path)
    features = ['SEVERITYCODE', 'ADDRTYPE', 'COLLISIONTYPE', 'PERSONCOUNT', 'PEDCOUNT', 'PEDCYLCOUNT',
               'VEHCOUNT', 'JUNCTIONTYPE', 'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 'ROADCOND', 'LIGHTCOND',
               'SPEEDING', 'HITPARKEDCAR']
    return df[features], features


def get_column_na_alt(col):
    fill_na_alt = {
        'SEVERITYCODE': 0,
        'ADDRTYPE': '',
        'INTKEY': 0,
        'COLLISIONTYPE': '',
        'PERSONCOUNT': 0,
        'PEDCOUNT': 0,
        'PEDCYLCOUNT': 0,
        'VEHCOUNT': 0,
        'JUNCTIONTYPE': '',
        'INATTENTIONIND': 'N',
        'UNDERINFL': '',
        'WEATHER': '',
        'ROADCOND': '',
        'LIGHTCOND': '',
        'SPEEDING': 'N',
        'HITPARKEDCAR': ''
    }
    return fill_na_alt[col]


def get_processed_dataframe(df, columns):
    column_map = {}
    features = ['ADDRTYPE', 'COLLISIONTYPE', 'PERSONCOUNT', 'PEDCOUNT', 'PEDCYLCOUNT',
                'VEHCOUNT', 'JUNCTIONTYPE', 'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 'ROADCOND', 'LIGHTCOND',
                'SPEEDING', 'HITPARKEDCAR']
    target = 'SEVERITYCODE'
    categorical_columns = ['COLLISIONTYPE', 'JUNCTIONTYPE', 'WEATHER', 'ROADCOND', 'LIGHTCOND']
    one_hot_columns = ['ADDRTYPE', 'INATTENTIONIND', 'UNDERINFL', 'SPEEDING', 'HITPARKEDCAR']
    for col in categorical_columns:
        df.fillna(get_column_na_alt(col), inplace=True)
        column_map[col] = {}
        unique_values = df[col].unique()
        for i, uv in enumerate(unique_values):
            column_map[col][uv] = i
    column_map['UNDERINFL'] = {'0': 'N',
                               '1': 'Y'}
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
    return df, column_map, features, target


def get_column_correlation(df):
    correlation = df.corr()
    sbs.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns)
    plt.show()


def apply_model(X_train, X_test, y_train, y_test, model):
    classifier = model
    classifier_model = classifier.fit(X_train, y_train)
    y_pred = classifier_model.predict(X_test)
    return get_accuracy_for_model(y_pred, y_test)


def get_accuracy_for_model(y_pred, y_test):
    return metrics.accuracy_score(y_test, y_pred)


def apply_model_and_get_accuracy(df):
    X = df.drop('SEVERITYCODE', axis=1)
    y = df.SEVERITYCODE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    model_list = [
        {
            'name': 'Decision Tree',
            'function': DecisionTreeClassifier(),
            'accuracy': 0
        },
        {
            'name': 'Gaussian NB',
            'function': GaussianNB(),
            'accuracy': 0
        },
        {
            'name': 'Nearest Neighbors',
            'function': NearestCentroid(),
            'accuracy': 0
        },
        {
            'name': 'Neural Network',
            'function': MLPClassifier(
                solver='lbfgs',
                alpha=1e-5,
                hidden_layer_sizes=(5, 2),
                random_state=1
            ),
            'accuracy': 0
        }
    ]
    for model in model_list:
        model['accuracy'] = apply_model(X_train, X_test, y_train, y_test, model['function'])
        print('Accuracy for ' + model['name'] + ': ', model['accuracy'])
    return model_list


def main():
    base_path = 'C:/Users/SumitKJ/PycharmProjects/AppliedDataScienceCapstone'
    file_path = base_path + '/data/Data-Collisions.csv'
    df_collision, features = get_dataframe(file_path)
    df_collision, column_map, features, target = get_processed_dataframe(df_collision, features)
    get_column_correlation(df_collision)
    model_list = apply_model_and_get_accuracy(df_collision)
    exit(0)


if __name__ == '__main__':
    main()

