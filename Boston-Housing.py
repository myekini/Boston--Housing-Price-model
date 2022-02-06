import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

PATH = "housing.csv"
califonia_data = pd.read_csv(PATH)

# index the columns in the dataset
califonia_data.columns

#setting prediction target(median house value)
y = califonia_data.median_house_value

#features to be considered for prediction
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'population', 'households', 'median_income', 'total_bedrooms', 'ocean_proximity',]

#setting Features
X = califonia_data[features]


# split data to get training and validation data
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=0)


# define function for prediting and evaluating our dataset
def score_all(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    mae = mean_absolute_error(val_y,preds)
    return mae


# determine columns with categorical variables 
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)

print("categorical variables")
print(object_cols)

# first approach is to drop these categorical variables 
train_X_dropped = train_X.select_dtypes(exclude=['object'])
val_X_dropped = val_X.select_dtypes(exclude=['object'])


# second approach is to ordinal encoding (make copy to avoid changing original data)
label_X_train = train_X.copy()
label_X_val = val_X.copy()

#apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(train_X[object_cols])
label_X_val[object_cols] = ordinal_encoder.fit_transform(val_X[object_cols])


# 3rd approach for handling categorical variables (apply one-hot encoder to each column with categorical data)
OH_encoder = OneHotEncoder(handle_unknown= 'ignore', sparse= False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]))
OH_cols_val = pd.DataFrame(OH_encoder.fit_transform(val_X[object_cols]))

#one hot encoding removed index. put it back
OH_cols_train.index = train_X.index
OH_cols_val.index = val_X.index

#removed categorical columns (will replace with one-hot encoding)
num_train_X = train_X.drop(object_cols, axis=1)
num_val_X = val_X.drop(object_cols, axis=1)

# add one - hot encoded columns to numerical features
OH_X_train = pd.concat([num_train_X, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_val_X, OH_cols_val], axis=1)



# First (1st) approach to handling missing data, get columns with missing data
col_with_missing = [col for col in label_X_train.columns
                   if label_X_train[col].isnull().any()]

train_X_reduced = label_X_train.drop(col_with_missing, axis=1)
val_X_reduced = label_X_val.drop(col_with_missing, axis=1)

print ("MAE from dropping columns")
score_all(train_X_reduced, val_X_reduced, train_y, val_y)


# second approach to handling missing data
my_imputer = SimpleImputer()
imputed_train_X = pd.DataFrame(my_imputer.fit_transform(label_X_train))
imputed_val_X = pd.DataFrame(my_imputer.transform(label_X_val))

# imputation removed column names; put them back
imputed_train_X.columns = label_X_train.columns
imputed_val_X.columns = label_X_val.columns

print("MAE from 2nd approach ")
score_all(imputed_train_X, imputed_val_X, train_y, val_y)



#third (3rd) approach to hadling missing data (making copy of the data to avoid changing originall data)
train_X_plus = OH_X_train.copy()
val_X_plus  = OH_X_valid.copy()

#find columns with missing data
col_with_missing = [col for col in OH_X_train.columns
                   if OH_X_train[col].isnull().any()]

#looping through the missing columns to add extra information
for col in col_with_missing:
    train_X_plus[col + '__was missing'] = train_X_plus[col].isnull()
    val_X_plus[col + '__was missing'] = val_X_plus[col].isnull()
    
#imputer
my_imputer = SimpleImputer()
imputed_train_X_plus = pd.DataFrame(my_imputer.fit_transform(train_X_plus))
imputed_val_X_plus = pd.DataFrame(my_imputer.fit_transform(val_X_plus))

#fix column names
imputed_train_X_plus.columns = train_X_plus.columns
imputed_val_X_plus.columns = val_X_plus.columns

print("MAE from 3rd approach ")
print(score_all(imputed_train_X_plus, imputed_val_X_plus, train_y, val_y))


#define the model with a random state equals 1 (old method 1)
califonia_model = DecisionTreeRegressor(random_state=1)
califonia_model.fit(imputed_train_X, train_y)
preds = califonia_model.predict(imputed_val_X)

def scoreall(val_y,preds):
    mae = mean_absolute_error(val_y,preds)
    return mae

scoreall(val_y, preds)


# making a better predictions with RandomForestRegressor and make predictions (old method)
califonia_model_2 = RandomForestRegressor(random_state=1)
califonia_model_2.fit(imputed_train_X, train_y)
preds_2 = califonia_model_2.predict(imputed_val_X)


#measuring the quality of the data
def scoreall(val_y,preds_2 ):
    mae = mean_absolute_error(val_y,preds_2)
    return mae

scoreall(val_y, preds_2)