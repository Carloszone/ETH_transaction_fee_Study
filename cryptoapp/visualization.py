import pandas as pd
import boto3
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error as mse
from cryptoapp.data import upload_file
import investpy


# settings
shift_n=1
previous = True
intercept=True
alpha=[0.01, 0.05, 0.1, 0.5, 1]
threshold=0.05
tokens = list(investpy.crypto.get_cryptos()['name'])
y_mark='Close.'

# function to slice df
def slice_df(df, y_label):
    token_index = df[y_label].ne(0).idxmax()
    transaction_index = df['old_ave_gas_fee'].ne(0).idxmax()
    ind = max(token_index, transaction_index)
    if token_index > transaction_index:
        ind = token_index
    else:
        ind = transaction_index
    sliced_df = df[ind:].copy()
    return sliced_df

# function to get df shift
def df_shift(dataset, y_label, shift_n = 1):
    df = dataset.copy()
    df = slice_df(df, y_label)
    new_col = y_label + '_p'
    df[new_col] = df[y_label].shift(periods= shift_n)
    df = df.dropna()
    return df

# function to split df into trainset and testset
def split_df(df):
    df = slice_df(df, y_label)
    partition = df.index[int(len(df)*0.8)]
    trainset = df[df.index <= partition]
    testset = df[df.index > partition]
    return trainset, testset

# function to normalize df
def df_preprocessing(df, type='standardize'):
    X = df.values
    if type == 'standardize':
        std_scaler = preprocessing.StandardScaler().fit(X)
        x_scaled = std_scaler.transform(X)
        res = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
        return res, std_scaler
    elif type == 'minmax':
        minmax_scaler = preprocessing.MinMaxScaler().fit(X)
        x_scaled = minmax_scaler.transform(X)
        res = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
        return res, minmax_scaler


# function to get data for modelling
def get_data(df, y_label, preprocess='standardize', intercept=True):
    # 01 split X and Y
    X = df.loc[:, df.columns != y_label]
    Y = df.loc[:, df.columns == y_label]

    # 02 preprocess
    scaler = None
    if preprocess == 'standardize':
        X, scaler = df_preprocessing(X, type='standardize')
    if preprocess == 'minmax':
        X, scaler = df_preprocessing(X, type='minmax')

    # 03 add constant term
    if intercept == True:
        X = sm.add_constant(X)
    return X, Y, scaler


def alpha_search(x, y, alpha=[0.01, 0.05, 0.1, 0.5, 1], type='lasso'):
    if type == 'ridge':
        ridge_cv = RidgeCV(alphas=alpha)
        model_cv = ridge_cv.fit(x, y.values.ravel())
        return model_cv.alpha_
    if type == 'lasso':
        lasso_cv = LassoCV(alphas=alpha)
        model_cv = lasso_cv.fit(x, y.values.ravel())
        return model_cv.alpha_


def liner_model(X, Y, type='lasso', alpha=None):
    model = sm.OLS(Y, X)
    results_fu = model.fit()
    Best_alpha = None
    if type == 'ridge':
        best_alpha = alpha_search(X, Y, alpha=alpha, type='ridge')
        model_ridge = model.fit_regularized(L1_wt=0, alpha=best_alpha, start_params=results_fu.params)
        ridge_result = sm.regression.linear_model.OLSResults(model, model_ridge.params, model.normalized_cov_params)
        return ridge_result, best_alpha
    elif type == 'lasso':
        best_alpha = alpha_search(X, Y, alpha=alpha, type='lasso')
        model_lasso = model.fit_regularized(L1_wt=1, alpha=best_alpha, start_params=results_fu.params)
        lasso_result = sm.regression.linear_model.OLSResults(model, model_lasso.params, model.normalized_cov_params)
        return lasso_result, best_alpha
    else:
        return results_fu, Best_alpha


@ignore_warnings(category=(ConvergenceWarning,UserWarning))
def backward_selection(dataframe, y_label, type='lasso', alpha=[0.01, 0.05, 0.1, 0.5, 1], threshold=0.05):
    df = dataframe.copy()
    X, Y, scaler = get_data(df, y_label=y_label)

    # create linear model
    model, best_alpha = liner_model(X, Y, type=type, alpha=alpha)

    # backward selection model
    # .1 get feature coef result
    res = list(model.pvalues)
    max_p = max(res)

    # .2 find the biggest coef and correlated feature name
    while max_p > threshold:
        ind = res.index(max_p)  # the index of max p value
        col = X.columns[ind]  # find the column name

        # .3 remove the feature from X
        X = X.drop(col, axis=1)
        # .4 build a new model
        if len(X.columns) == 0:
            print('all features have been removed, return the last avaiable model')
            return model, X, best_alpha, scaler
        model, best_alpha = liner_model(X, Y, type=type, alpha=alpha)
        res = list(model.pvalues)
        max_p = max(res)

    # return result
    return model, X, best_alpha, scaler

# function to load model from a file:
def load_model():
    bucket_name = 'carlos-cryptocurrency-research-project'
    model_path = 'data/model/backward_selection_lasso_regression_model.pickle'
    model_name = 'backward_selection_lasso_regression_model.pickle'
    s3_r = boto3.resource("s3")
    s3_r.meta.client.download_file(bucket_name, model_path, model_name)
    model = sm.load(model_name)
    return model

class backward_selection_model:
    def __init__(self, df, y_label, type='lasso', alpha=[0.01, 0.05, 0.1, 0.5, 1], threshold=0.05):
        self.original_df = df.copy()
        self.dataframe = df_shift(df, y_label)
        self.y_label = y_label
        self.type = type
        self.alpha = alpha
        self.threshold = threshold
        model, X, best_alpha, scaler = backward_selection(self.dataframe, y_label=self.y_label, type=self.type,
                                                          alpha=self.alpha, threshold=self.threshold)
        self.model = model
        self.X = X
        self.best_alpha = best_alpha
        self.scaler = scaler

    def get_model(self):
        return self.model

    def get_final_features(self):
        return self.X.columns

    def get_best_alpha(self):
        return self.best_alpha

    def get_scaler(self):
        return self.scaler

    def get_prediction(self):
        model = self.get_model()
        res = model.get_prediction()
        summary = res.summary_frame()
        upper = res.summary_frame()["obs_ci_upper"]
        lower = res.summary_frame()["obs_ci_lower"]
        prediction = model.fittedvalues
        true_y = self.dataframe[self.y_label].values
        date = self.dataframe.index
        print('Date lenght: ', len(date))
        print('Prediction length: ', len(prediction))
        print('upper lenght: ', len(upper))
        print('lower lenght: ', len(lower))
        print('ture lenght: ', len(true_y))
        print('df lenght', len(self.dataframe))
        print('dx lenght', len(self.X))
        print(self.X)
        print(summary)


        Ys = pd.DataFrame({'Date': date, 'True_v': true_y, 'Prediction': prediction, 'Upper_p': upper, 'Lower': lower})
        Ys['Targets'] = self.y_label
        Ys = Ys.reset_index(drop = True)
        return Ys

    def get_mse(self):
        prediction = self.get_prediction()
        target_df = self.df.copy()
        target_y = target_df.loc[:, target_df.columns == self.y_label].values.ravel()
        return mse(target_y, prediction)

    def get_coef_df(self):
        return pd.DataFrame({'coef': self.model.params, 'P-value': self.model.pvalues})

    def save_model(self, filename = "backward_selection_lasso_regression_model.pickle",
                   bucket_path = 'data/model/'):
        model = self.model
        model.save(filename)
        upload_file(filename, bucket_path = bucket_path)


#test_path = 'https://raw.githubusercontent.com/Carloszone/Cryptocurrency_Research_project/main/datasets/test.csv'
#df = pd.read_csv(test_path, parse_dates = ['Date']).set_index('Date')

import time
start_time = time.time()
res = []
count = 1
total = len(tokens)
for token in tokens:
    print(f'Now processing {token} {count}/{total}')
    var_list = ['S&P 500', 'Nasdaq', 'DJ Composite', 'Gold', 'Copper', 'Silver', 'Crude Oil WTI', 'Natural Gas']
    count += 1
    var_list.append(token)
    y_label = y_mark + token
    try:
        df = pd.read_csv('bc_test.csv')
        test = backward_selection_model(df, y_label=y_label)
        prediction = test.get_prediction()
        res.append(prediction)
    except:
        print(f'cannot find the token ({token}) record')
    print("--- %s seconds ---" % (time.time() - start_time))
pres = pd.concat(res, ignore_index=True)
pres.to_csv('predictions.csv')
print('final mission completed')
print(f'cannot find the token ({token}) record')

def join_df(var_list):
    transaction_df = pd.read_csv()
    p1_df = pd.read_csv()
    p2_df = pd.read_csv()
    p3_df = pd.read_csv()
    index_df = pd.read_csv()
    commodity_df = pd.read_csv()