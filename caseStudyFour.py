import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)

################################################
# Exploratory Data Analysis (EDA)
################################################
#region EDA

# ------------------------------
# 1. Veri Yükleme ve Birleştirme
# ------------------------------
def load_train():
    df = pd.read_csv("datasets/train.csv")
    return df

def load_test():
    df = pd.read_csv("datasets/test.csv")
    return df

train_df = load_train()
test_df = load_test()

test_df["SalePrice"] = np.nan
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

print(f"Train Seti Boyutu: {train_df.shape}")
print(f"Test Seti Boyutu: {test_df.shape}")
print(f"Birleştirilmiş Veri Seti Boyutu: {df.shape}")

# ------------------------------
# 2. Değişken Türlerini Belirleme
# ------------------------------
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtype != 'O']
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtype == 'O']

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Obversations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# ------------------------------
# 3. Sayısal ve Kategorik Değişkenlerin Analizi
# ------------------------------
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    }))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe, palette="viridis")
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# ------------------------------
# 4. Kategorik Değişkenler ile Hedef Değişken Analizi
# ------------------------------
def target_summary(dataframe, target_col, categorical_col):
    print(pd.DataFrame(dataframe.groupby(categorical_col)[target_col].mean()), end="\n\n\n")

for col in cat_cols:
    target_summary(df, "SalePrice", col)

# ------------------------------
# 5. Aykırı Gözlem Analizi
# ------------------------------
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# ------------------------------
# 6. Eksik Veri Analizi
# ------------------------------
def missing_values_table(dataframe, na_name=False):
    na_contain_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_contain_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_contain_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_contain_columns

na_contain_columns = missing_values_table(df, na_name=True)

#endregion

################################################
# Feature Engineering
################################################
#region Feature Engineering

# ------------------------------
# 1. Yeni Değişkenler Ekleyelim
# ------------------------------
df["TOTAL_SF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]
df["TOTAL_BATH"] = df["FullBath"] + df["HalfBath"] * 0.5 + df["BsmtFullBath"] + df["BsmtHalfBath"] * 0.5
df["HOUSE_AGE"] = df["YrSold"] - df["YearBuilt"]
df["SINCE_REMODEL"] = df["YrSold"] - df["YearRemodAdd"]

df["HAS_GARAGE"] = df["GarageYrBlt"].apply(lambda x: 1 if x > 0 else 0)
df["HAS_POOL"] = df["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
df["HAS_FIREPLACE"] = df["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)
df["HAS_BASEMENT"] = df["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)

# ------------------------------
# 2. Rare Encoding (Seyrek Kategorileri Gruplama)
# ------------------------------
def rare_encoder(dataframe, rare_perc=0.01):
    rare_columns = [col for col in dataframe.columns if dataframe[col].dtype == "O"
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).sum() > 0]

    for var in rare_columns:
        tmp = dataframe[var].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[var] = dataframe[var].replace(rare_labels, "Rare")

    return dataframe

df = rare_encoder(df)

# ------------------------------
# 3. Encoding İşlemleri
# ------------------------------
binary_cols = ["CentralAir", "Street", "PavedDrive", "HAS_GARAGE", "HAS_POOL", "HAS_FIREPLACE", "HAS_BASEMENT"]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["SalePrice"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

# ------------------------------
# 4. Scaling (Ölçekleme)
# ------------------------------
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#endregion

################################################
# Modeling
################################################
#region Modeling

# --- KATEGORİK SÜTUNLARI SAYISALA DÖNÜŞTÜRME FONKSİYONU ---
def encode_categorical(dataframe):
    cat_cols = dataframe.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        dataframe[col] = le.fit_transform(dataframe[col].astype(str))
    return dataframe

# Train ve test setlerini oluştururken kopyalarını alıyoruz
df_train = df[df["SalePrice"].notnull()].copy()
df_test = df[df["SalePrice"].isnull()].copy()

# Kalan kategorik sütunları sayısallaştırıyoruz
df_train = encode_categorical(df_train)
df_test = encode_categorical(df_test)

# Bağımlı ve bağımsız değişkenleri belirleme
y = df_train["SalePrice"].astype(int)
X = df_train.drop(["SalePrice"], axis=1)

# Train-test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#region Model Training & Evaluation
cart_model = DecisionTreeRegressor(random_state=42)
cart_model.fit(X_train, y_train)

y_pred = cart_model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"R^2: {r2_score(y_test, y_pred)}")
#endregion

#region Hyperparameter Optimization
cart_params = {
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": [None, 'auto', 'sqrt', 'log2'],
}

cart_best_grid = GridSearchCV(
    cart_model,
    cart_params,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='neg_mean_squared_error'
)
cart_best_grid.fit(X_train, y_train)

print(f"En iyi hiperparametreler: {cart_best_grid.best_params_}")

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X_train, y_train)
print(f"Final Model Parametreleri: {cart_final.get_params()}")

cv_results = cross_validate(
    cart_final,
    X_train,
    y_train,
    cv=5,
    scoring=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
)

print(f"CV R^2: {cv_results['test_r2'].mean()}")
print(f"CV MSE: {-cv_results['test_neg_mean_squared_error'].mean()}")
print(f"CV MAE: {-cv_results['test_neg_mean_absolute_error'].mean()}")

#MSE: 0.2226027397260274
#MAE: 0.2089041095890411
#R^2: 0.6477357089829251

"""CV R^2: 0.5370072708974116
CV MSE: 0.20651982112303297
CV MAE: 0.2090158584275988"""


#endregion



#region Feature Importances
def plot_importance(model, features, num=None, save=False):
    if num is None:
        num = len(features.columns)

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})

    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False).head(num)
    )
    plt.title("Features")
    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig("importances.png")

plot_importance(cart_final, X_train, num=5, save=True)

#endregion

#endregion
