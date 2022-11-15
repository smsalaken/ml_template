from textwrap import fill
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
import joblib


from sklearn.datasets import load_wine

# set a random number
randseed = 42389

# load data
data = load_wine()
x = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.DataFrame(data.target, columns = ['wine'])



# make preprocessing pipeline
# since simpleImputer do not return feature names, we need to add a method
SimpleImputer.get_feature_names_out = (lambda self, names = None: self.feature_names_in_)


# define a number of pipeline options
numeric_pipeline_with_impute = make_pipeline(SimpleImputer(strategy= "constant", fill_value= 0), StandardScaler())
numeric_pipeline_without_impute = make_pipeline(StandardScaler())
noncount_impute_pipeline = SimpleImputer(strategy="constant", fill_value= -999)
count_impute_pipeline = SimpleImputer(strategy="constant", fill_value= -1)
categorical_pipeline = OneHotEncoder()
passthrough_pipeline = make_pipeline("passthrough")




# make up a list of different features for toy example
numeric_features = ['alcohol', 'malic_acid', 'ash']
categorical_features = ['alcalinity_of_ash', 'magnesium', 'hue']
count_impute_features = ['total_phenols', 'flavanoids', 'nonflavanoid_phenols']
count_nonimpute_features = ['proanthocyanins', 'color_intensity']
remaining_features = ['od280/od315_of_diluted_wines', 'proline']

# make the custom transformer for preprocessing
# will be fit only on the training data
transformer = make_column_transformer(
    (numeric_pipeline_with_impute, numeric_features),
    (categorical_pipeline, categorical_features),
    (count_impute_pipeline, count_impute_features),
    (noncount_impute_pipeline, count_nonimpute_features),
    (passthrough_pipeline, remaining_features)

)


# run kfoldcv
rskf = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = randseed)

for train_index, test_index in rskf.split(x,y):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[test_index], y.iloc[test_index]

    # fit the preprocessing pipeline on train dataset
    transformer.fit(x_train)

    # specify feature names after preprocessing
    # will need this in the feature importance exploration
    x_train_preprocessed = pd.DataFrame(transformer.transform(x_train), columns = transformer.get_feature_names_out())
    x_test_preprocessed = pd.DataFrame(transformer.transform(x_test), columns = transformer.get_feature_names_out())

    # train classifier
    clf = RandomForestClassifier(
        class_weight = 'balanced',
        n_estimators = 100,
        max_depth = 5,
        random_state = randseed
    )

    clf.fit(x_train_preprocessed, y_train)

    # predict and evaluate
    y_pred =  clf.predict(x_test_preprocessed)
    y_pred_proba = clf.predict_proba(x_test_preprocessed)[:,1]

    acc = accuracy_score(y_test, y_pred)
    pcc = precision_score(y_test, y_pred)
    rcc = recall_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print("Accuracy: {0}, AUC: {1}, Precision: {2}, Recall: {3}".format(
        acc, auc_score, pcc, rcc
    ))
    print(classification_report(y_test, y_pred))

    # save model
    transformer_file_name = "transformer_acc_{}_auc_{}_precision_{}_recall_{}.joblib".format(
        round(acc,2), round(auc_score,2), round(pcc,2), round(rcc,2)
    )
    model_file_name = "model_acc_{}_auc_{}_precision_{}_recall_{}.joblib".format(
        round(acc,2), round(auc_score,2), round(pcc,2), round(rcc,2)
    )
    joblib.dump(transformer, transformer_file_name)
    joblib.dump(clf, model_file_name)