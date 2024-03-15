#Demand Forecasting

#Trying to obtain forecasts for 3 months ahead in terms of store-product breakdown

import pandas as pd
import warnings
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns

#pd.set_option('display.max_columns', None)
#pd.set_option('display_width', 500)
warnings.filterwarnings('ignore')

#getting information function for dataframe
def check_df(df, head=5):
    print('########################## Shape ############################')
    print(df.shape)
    print('########################## Type #############################')
    print(df.dtypes)
    print('########################## Head #############################')
    print(df.head(head))
    print('########################## Tail #############################')
    print(df.tail(head))
    print('##########################  NA  #############################')
    print(df.isnull().sum())
    print('######################  Quantiles  ##########################')
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


#loading dataset
train = pd.read_csv('/Users/handecavsi/Desktop/Miuul/ML forcasting/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('/Users/handecavsi/Desktop/Miuul/ML forcasting/demand_forecasting/test.csv', parse_dates=['date']) 

#number of variables requested from us/ id and sales
sample_sub = pd.read_csv('/Users/handecavsi/Desktop/Miuul/ML forcasting/demand_forecasting/sample_submission.csv') 
#print(sample_sub)

#combining train and test datasets
data_combined = pd.concat([train,test], sort=False)
#print(data_combined)

############# EDA ###################
min_date = data_combined['date'].min() #2013-01-01 00:00:00
max_date = data_combined['date'].max() #2018-03-31 00:00:00

#print(min_date)
#print(max_date)

check_df(data_combined) #Since the train and test dataset were combined, NAN values were created for sales and ids.

#Numbers of unique item and store
print(data_combined[['store']].nunique()) #10
print(data_combined[['item']].nunique()) #50

#Are the same 50 products sold in every store?
print(data_combined.groupby(['store'])['item'].nunique())

################Output##################
#store  item
#  1     50
#  2     50
#  3     50
#  4     50
#  5     50
#  6     50
#  7     50
#  8     50
#  9     50
#  10    50
########################################

#How many of the 50 unique products were sold in each store?
print(data_combined.groupby(['store', 'item']).agg({'sales':['sum']}))

#######Output############
#store item   sales_sum       
#1     1      36468.0
#      2      97050.0
#      3      60638.0
#      4      36440.0
#      5      30335.0
#...              ...
#10    46    120601.0
#      47     45204.0
#      48    105570.0
#      49     60317.0
#      50    135192.0
########################

#Sales statistics according to store and item breakdown
print(data_combined.groupby(['store', 'item']).agg({'sales':['sum', 'mean', 'median', 'std']})) 


#FEATURE ENGINEERING#
print(data_combined.head()) #date  store  item  sales  id

#Time-series data concepts such as seasonality, trend
# should be produced as variables for the machine learning algorithm.
def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df['is_wknd'] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df
    
    
data_combined = create_date_features(data_combined)

#print(data_combined)

#Sales statistics according to store,item, and month breakdown
print(data_combined.groupby(['store', 'item','month']).agg({'sales':['sum', 'mean', 'median', 'std']})) 

#noise generation to prevent overfitting
def random_noise(df):
    return np.random.normal(scale=1.6, size=(len(df),))
    
#Lag/Shifted Features#

#Ordering according to store,item, and date
data_combined.sort_values(by=['store','item','date'], axis=0, inplace=True)

#print(pd.DataFrame({"sales": data_combined['sales'].values[0:10],
            #"lag1": data_combined['sales'].shift(1).values[0:10],
            #"lag2": data_combined['sales'].shift(2).values[0:10],
            #"lag3": data_combined['sales'].shift(3).values[0:10],
            #"lag4": data_combined['sales'].shift(4).values[0:10]}))

############# OUTPUT ###############
#    yt    yt-1  yt-2  yt-3  yt-4
#   sales  lag1  lag2  lag3  lag4
#0   13.0   NaN   NaN   NaN   NaN
#1   11.0  13.0   NaN   NaN   NaN
#2   14.0  11.0  13.0   NaN   NaN
#3   13.0  14.0  11.0  13.0   NaN
#4   10.0  13.0  14.0  11.0  13.0
#5   12.0  10.0  13.0  14.0  11.0
#6   10.0  12.0  10.0  13.0  14.0
#7    9.0  10.0  12.0  10.0  13.0
#8   12.0   9.0  10.0  12.0  10.0
#9    9.0  12.0   9.0  10.0  12.0
###################################
#Note: A time series data is affected by the values before it!
#It breaks the univariate time-dependent view and adds previous values as dependent features and creates tabular data.

#print(data_combined.groupby(['store','item'])['sales'].head())
#print(data_combined.groupby(['store','item'])['sales'].transform(lambda x: x.shift(1)))


#By entering various lag amounts, previous values are added as dependent variables.
#Noise is added to the new dataframe to prevent overfitting.
def lag_features(df,lags):
    for lag in lags:
        df['sales_lag_' + str(lag)] = df.groupby(['store','item'])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(df)
    return df

#lags bring data from 3 months/90 days ago. NAN values can be evaluated accordingly.
data_combined = lag_features(data_combined, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

#check_df(data_combined)


## Rolling Mean Features ##


# The rolling method takes the average of previous values, including itself. 
# This is not productive for examining past value effects. 
# Also, it is not effective because there will be no value tomorrow.

#print(pd.Dataframe({'sales': data_combined['sales'].values[0:10],
#            'roll2': data_combined['sales'].rolling(window=2).mean().values[0:10],
#            'roll3': data_combined['sales'].rolling(window=3).mean().values[0:10],
#            'roll5': data_combined['sales'].rolling(window=5).mean().values[0:10]
#}))

#For this reason, it is healthier to take rolling after using lag/shift:
#print(pd.DataFrame({'sales': data_combined['sales'].values[0:10],
            #'roll2': data_combined['sales'].shift(1).rolling(window=2).mean().values[0:10],
            #'roll3': data_combined['sales'].shift(1).rolling(window=3).mean().values[0:10],
            #'roll5': data_combined['sales'].shift(1).rolling(window=5).mean().values[0:10]
#}))

def roll_mean_feature(df, windows):
    for window in windows:
        df['sales_roll_mean_' + str(window)] = df.groupby(['store', 'item'])['sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10).mean()) # 'win_type' kaldırıldı
    return df



# It tries to reflect data from 1 year and 1.5 years ago.
data_combined = roll_mean_feature(data_combined, [365, 546])


# Exponentially Weighted Features #

def ewm_features(df, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            df['sales_ewm_alpha_' + str(alpha).replace('.','') + 'lag_' + str(lag)] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return df

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]
data_combined = ewm_features(data_combined, alphas, lags)
#check_df(data_combined)


#One Hot Encoding#

data_combined = pd.get_dummies(data_combined, columns = ['store','item','day_of_week','month'])

#check_df(data_combined)

#Converting Sales to Log(1+Sales)#

#Since the dependent variable in LightGBM is numerical, 
# the Standardization process is needed because the duration of the Iteration operations is 
# based on the interaction of the predicted values and the residuals. 
# The aim is to reduce iteration and train time.

data_combined['sales'] = np.log1p(data_combined['sales'].values)

#Custom Cost Function#

#MAE, MSE, RMSE, SSE => Metrics used to evaluate errors and success.
#A customized custom cost function for LightGBM will be tried.

#MAE: Mean Absolute Error
#MAPE: Mean Absolute Percentage Error
#sMAPE: Symmetric Mean Absolute Percentage Error (Adjusted MAPE)


#sMAPE function
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200*np.sum(num/denom)) / n
    return smape_val

#sMAPE function for lgbm
def lgbm_smape(preds, train_data):
    labels = train_data.get_label() #gel_laber refers to dependent value in time series data/ Real data.
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

#Time-based validation sets#

train = data_combined.loc[(data_combined['date']<'2017-01-01'), :]
val = data_combined.loc[(data_combined['date']>='2017-01-01') & (data_combined['date']<'2017-04-01'), :]

#Feature selection 
cols = [col for col in train.columns if col not in ['date','id','sales','year']]

Y_train = train['sales'] #dependent value
X_train = train[cols] #independent values

Y_val = val['sales'] #dependent value
X_val = val[cols] #independent values

#print(Y_train.shape)
#print(X_train.shape)
#print(Y_val.shape)
#print(X_val.shape)


#Time series based LightGBM model
# !pip install lightgbm 
# conda install lightgbm

#LightGBM Parameters:
lgb_params = {'num_leaves': 10,
            'learning_rate': 0.2,
            'feature_fraction': 0.8,
            'max_depth': 5,
            'verbose': 0,
            'num_boost_round': 1000,
            'early_stopping_round': 200,
            'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square_loss, mean_squared_error, mse, regression_l2, regression
# rmse: root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# learning_rate: shrinkage_rate, eta
# feature_fraction: random number of variable at every iteration
# num_boost_round: n_estimators, number of boosting iterations. least around 10000 to 15000
# early_stopping_rounds: If the error does not decrease after a certain value, stop the training. Let's not memorize, let's learn:) 
# Shortens train time, prevents overfitting
# nthread: num_thread, nthread, nthreads, n_jobs

#Create specific data for lightGBM
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain, 
                valid_sets=[lgbtrain,lgbval],
                num_boost_round=lgb_params['num_boost_round'],
                feval=lgbm_smape)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
smape(np.expm1(y_pred_val), np.expm1(Y_val)) # sMAPE: 13.83

#Feature Importance#

#-Feature importance is the value it provides in error after division in a tree algorithm.
#-It is the entropy change before and after the split when the feature is used and divided according to certain split points.
#-When we examine the dependent variable after the division process, we have the estimated values for the dependent variable.
#-Entropy values (SSE change and MSE change) related to the differences between the actual values and the predicted values give the gain.
#-Feature importance can be determined by gain or split.
#-If split and gain are inversely proportional, these values are normalized between 0 and 1, 
# and a single feature importance can be obtained by multiplying the normalized values.

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                            'split': model.feature_importance('split'),
                            'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10,10))
        sns.set(font_scale = 1)
        sns.barplot(x='gain',y='feature', data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
plot_lgb_importances(model, num=30)
#plot_lgb_importances(model, num=30, plot = True)


# Final Model #

train = data_combined.loc[~data_combined.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = data_combined.loc[data_combined.sales.isna()]
X_test = test[cols]

#LightGBM Parameters:
lgb_params = {'num_leaves': 10,
            'learning_rate': 0.2,
            'feature_fraction': 0.8,
            'max_depth': 5,
            'verbose': 0,
            'num_boost_round': model.best_iteration,
            'nthread': -1}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params,lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

#print(test_preds)

#Submission File#
#test.head() #format requested from us
submission_df = test.loc[:, ['id','sales']]
submission_df['sales'] = np.expm1(test_preds) ##converting logarithmic values to normal
submission_df['id'] = submission_df.id.astype(int) 

submission_df.to_csv('submission_demand.csv', index = False)