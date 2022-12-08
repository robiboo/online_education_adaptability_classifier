# Robert Panerio
# Data Mining 315 - Adaptability Level of students using Naive Bayes
from pandas import read_csv, get_dummies
from math import log, pow

def cond_prob(train, ft):
    train_nb = train
    cond_prob = {}
    alpha = 0.00000001
    num_features = 13

    # a list of names of features; not including the adaptivity level
    ft_names = list(train_nb.columns)[1:]

    # prints the names of the features
    for ft in ft_names:
        cond_prob_ft = list(train_nb.groupby(['Adaptivity Level', ft]).size().unstack(fill_value=0).stack())
        cond_prob[ft] = (train_nb.groupby(['Adaptivity Level', ft]).size().unstack(fill_value=0).stack()) + alpha


def base_probability(train):
    alpha = 0.00000001
    num_features = 13
    temp_train = train
    len_train = len(temp_train)

    base_prob = (temp_train.groupby('Adaptivity Level').size().add(alpha)).div(len_train + (alpha + num_features))
    print(base_prob)
    adaptivity_level = ['High', 'Low', 'Moderate']
    for level in adaptivity_level:
        base_prob[level] = log(base_prob[level], 2)
    print(base_prob)
    return base_prob

def niave_bayes(train, test):
    final_predict = {}
    train_nb = train
    test_nb = test

    # calculate the base probabilty for each adaptivity level
    base_prob = base_probability(train_nb)


    for row in test_nb.iterrows():
        print(row)


def data_input():
    # read the data set
    data = read_csv('./students_adaptability_level_online_education.csv')

    # transform categorical features into numerical except the target feature
    data = get_dummies(data=data, columns=['Gender', 'Age', 'Education Level', 'Institution Type', 'IT Student', 'Location',\
                                           'Load-shedding', 'Financial Condition', 'Internet Type', 'Network Type',\
                                           'Class Duration', 'Self Lms', 'Device'])
    # fill NaN values with 0
    data = data.fillna(0)

    train_data = data.sample(frac=0.7)
    test_data = data.drop(train_data.index)
    niave_bayes(train_data, test_data)

    print(data)
    print('# of columns',len(data.columns))
    print('# of rows',len(data))

def main():
    data_input()


if __name__ == '__main__':
    main()