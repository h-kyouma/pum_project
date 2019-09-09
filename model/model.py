import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
from pandas.plotting import scatter_matrix
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ATTRIBS = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21',
           'mu12', 'mu03', 'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03', 'contour_areas', 'contour_lengths',
           'convex_hull_areas', 'convex_hull_lengths', 'rectangle_rotations', 'cnt_area_cnt_len_ratio',
           'hull_area_hull_len_ratio', 'hull_area_cnt_area_ratio', 'hull_len_cnt_len_ratio', 'leftmost_point',
           'leftmost_point_vect_mag', 'rightmost_point', 'rightmost_point_vect_mag', 'topmost_point',
           'topmost_point_vect_mag', 'bottommost_point', 'bottommost_point_vect_mag', 'vertical_height',
           'horizontal_height', 'vertical_to_horizontal_ratio']


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


def stratified_split(df, info=False):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df['label']):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]
        train_set.to_csv('train_data.csv')
        test_set.to_csv('test_data.csv')

        if info is not False:
            print('whole dataset\n', df['label'].value_counts()/len(df), '\n')
            print('train dataset\n', train_set['label'].value_counts() / len(train_set), '\n')
            print('test dataset\n', test_set['label'].value_counts() / len(test_set), '\n')


def show_histogram(df):
    df.hist(bins=100)
    plt.show()


def save_data_info(df, filename):
    buffer = io.StringIO()
    df.info(buf=buffer)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(buffer.getvalue())


def calculate_correlations(df):
    if not os.path.exists('correlations'):
        os.makedirs('correlations')
    corr_matrix = df.corr()
    corr_matrix.to_excel('correlations/correlation.xlsx')
    for column in df:
        if 'Unnamed' not in column:
            buffer = io.StringIO()
            corr_matrix[column].sort_values(ascending=False).to_string(buf=buffer)
            with open('correlations/{}.txt'.format(column), 'w', encoding='utf-8') as f:
                f.write(buffer.getvalue())


def show_scatter_plots(df, attributes=None):
    if attributes is None:
        attributes = ['vertical_to_horizontal_ratio', 'cnt_area_cnt_len_ratio', 'hull_area_hull_len_ratio',
                      'hull_len_cnt_len_ratio']
    scatter_matrix(df[attributes])
    plt.show()


def main():
    stratified_split(pd.read_csv('data.csv', index_col=0))

    data = pd.read_csv('train_data.csv', index_col=0)
    train_set = data.copy()

    # save_data_info(train_set, 'train_data_info.txt')
    # calculate_correlations(train_set)
    # show_histogram(data)
    # show_scatter_plots(data)

    y = train_set['label']
    X = train_set.drop(columns='label')

    data_preprocessor = Pipeline(steps=[
        ('selector', DataFrameSelector(ATTRIBS)),
        ('std_scaler', StandardScaler())
    ])

    X_prepared = data_preprocessor.fit_transform(X)
    joblib.dump(data_preprocessor, 'data_preprocessor.pkl')

    model_selector = Pipeline(steps=[
        ('pca', PCA(n_components=0.99)),
        ('svm_clf', SVC(kernel='rbf', gamma='scale'))
    ])

    param_grid = {
        'svm_clf__C': [1, 5, 10, 15, 20]
    }

    grid_search = GridSearchCV(model_selector, param_grid, cv=5, verbose=100)
    grid_search.fit(X_prepared, y)

    print('Best parameters:', grid_search.best_params_)

    final_model = grid_search.best_estimator_
    joblib.dump(final_model, 'best_model.pkl')

    test_data = pd.read_csv('test_data.csv')
    test_set = test_data.copy()
    y_test = test_set['label']
    X_test = test_set.drop(columns='label')

    X_test_prepared = data_preprocessor.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)


if __name__ == '__main__':
    main()
