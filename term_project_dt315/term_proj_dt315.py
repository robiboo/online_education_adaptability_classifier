# Robert Panerio
# Data Mining 315 - Adaptability Level of students using Naive Bayes
from pandas import read_csv, get_dummies, DataFrame
import pandas as pd
from math import log, pow
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel

def cond_probability(train):
    train_nb = train
    cond_prob = {}
    alpha = 0.00000001
    num_features = 35

    adaptivity_level = ['High', 'Low', 'Moderate']

    # a list of names of features; not including the adaptivity level
    ft_names = list(train_nb.columns)[1:]

    # prints the names of the features
    for ft in ft_names:
        cond_prob_ft = list(train_nb.groupby(['Adaptivity Level', ft]).size().unstack(fill_value=0).stack())
        cond_prob[ft] = (train_nb.groupby(['Adaptivity Level', ft]).size().unstack(fill_value=0).stack()) + alpha

        count = 0
        for level in adaptivity_level:
            x = cond_prob_ft[count] + cond_prob_ft[count +1]
            # print(x)
            for value in range(2):
                cond_prob[ft][level][value] = cond_prob[ft][level][value] / (x + (alpha * num_features))
                cond_prob[ft][level][value] = log(cond_prob[ft][level][value], 2)

            count += 2
    return cond_prob

def base_probability(train):
    alpha = 0.00000001
    num_features = 35
    temp_train = train
    len_train = len(temp_train)

    base_prob = (temp_train.groupby('Adaptivity Level').size().add(alpha)).div(len_train + (alpha + num_features))
    adaptivity_level = ['High', 'Low', 'Moderate']
    for level in adaptivity_level:
        base_prob[level] = log(base_prob[level], 2)

    return base_prob

def niave_bayes_own(train, test):
    final_predict = {'Adaptivity Level': [], 'Prediction': [], 'Probability': []}
    train_nb = train
    test_nb = test

    # calculate the base probabilty for each adaptivity level
    base_prob = base_probability(train_nb)
    cond_prob = cond_probability(train_nb)

    adaptivity_level = ['High', 'Low', 'Moderate']
    features_names = list(train_nb.columns)
    test_adaptivity_level = test_nb['Adaptivity Level']

    accuracy_count = 0
    for index, row in test_nb.iterrows():
        nb_calc = {}
        inv_nb = {}
        list_row = list(row)

        for level in adaptivity_level:
            nb_calc[level] = base_prob[level]
            inv_nb[level] = (1 - base_prob[level])

            for x in range(1, len(list_row)):
                nb_calc[level] = nb_calc[level] + cond_prob[features_names[x]][level][list_row[x]]

        for level2 in adaptivity_level:
            nb_calc[level2] = pow(2, nb_calc[level2])

        max_num = 0
        pred_level = ''
        for key, values in nb_calc.items():
            if values >= max_num:
                max_num = values
                pred_level = key

        inv_num = 0
        for key, values in nb_calc.items():
            if values != max_num:
                inv_num = inv_num + values

        res = max_num/(max_num + inv_num)
        final_predict['Adaptivity Level'].append(list_row[0])
        final_predict['Prediction'].append(pred_level)
        final_predict['Probability'].append(res)
        if list_row[0] != pred_level:
            accuracy_count += 1

    fin = DataFrame.from_dict(final_predict)
    # print(fin.to_string(index=False))
    print(f'    # of Right Predictions {len(test_nb) - accuracy_count} out of {len(test_nb)} tests')
    print(f"    My Naive Bayes: {((len(test_nb) - accuracy_count)/len(test_nb))*100}\n")

def data_preprocessing(data):
    new_data = data
    data_encode = {}
    for feature in new_data.columns:
        data_encode[feature] = LabelEncoder()
        new_data[feature] = data_encode[feature].fit_transform(new_data[feature])
    return new_data
def random_forest(nt_train, nt_test, t_train, t_test):
    rfc = RandomForestClassifier(random_state=30)
    ft = SelectFromModel(rfc)
    ft.fit(nt_train, t_train)
    rfc.fit(nt_train, t_train)
    pred = rfc.predict(nt_test)
    print(f"    Random Forest Classifier: {accuracy_score(t_test, pred)*100}")
    ft_lst = list(nt_train.columns[(ft.get_support())])
    print(f"    Features Selected:")
    for feat in ft_lst:
        print(f"            {feat}")


def naive_bayes(nt_train, nt_test, t_train, t_test):
    nv = GaussianNB()
    nv.fit(nt_train, t_train)
    pred = nv.predict(nt_test)
    acc = accuracy_score(t_test, pred)
    print(f'    # of Right Predictions {int(acc*len(t_test))} out of {len(t_test)} tests')
    print(f"    Naive Bayes: {acc*100}")
def data_input():
    # read the data set
    data = read_csv('./students_adaptability_level_online_education.csv')

    # transform categorical features into numerical except the target feature
    data_num = get_dummies(data=data, columns=['Gender', 'Age', 'Education Level', 'Institution Type', 'IT Student', 'Location',\
                                           'Load-shedding', 'Financial Condition', 'Internet Type', 'Network Type',\
                                           'Class Duration', 'Self Lms', 'Device'])

    train_data = data_num.sample(frac=0.699999999)
    test_data = data_num.drop(train_data.index)

    print("\nOwn Naive Bayes vs Scikit Learn's Naive Bayes")
    niave_bayes_own(train_data, test_data)

    # pre-processed the data instead of using get_dummies
    new_data = data_preprocessing(data)

    # get all non target features
    non_target = new_data.drop(columns="Adaptivity Level")

    # target features
    target = data["Adaptivity Level"]

    # randomly split data into train and test sets
    non_target_train, non_target_test, target_train, target_test = train_test_split(non_target, target, test_size=0.3,
                                                                                    random_state=30)

    # using sklearn classifier models
    naive_bayes(non_target_train, non_target_test, target_train, target_test)

    print('\nUsing Randon Forest Classifier:')
    random_forest(non_target_train, non_target_test, target_train, target_test)
    print('')
def main():
    data_input()


if __name__ == '__main__':
    main()