# 1. import Libraries
import streamlit as st
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score

st.set_page_config(page_title='Machine Learning HyperPrameters Tuning App', page_icon=':panda_face:', layout='wide')
tab1, tab2 = st.tabs(['Model evaluation report','About App'])

with tab1:

    # 2. Page Outlook

    st.title('HyperParameters Tuning App')
    st.text('--------------------------Gives Best Fit----------------------')

    st.markdown("""      
    # Explore different classifier and datasets
    """)
    st.text('Which One Is Best?')
    dataset_name = st.sidebar.selectbox(
        'Select Dataset',
        ('Digits', 'Iris', 'Breast Cancer', 'Wine')
    )

    st.write(f"## Dataset using : {dataset_name} Dataset")


    classifier_name = st.sidebar.selectbox(
        'Select classifier',
        ('KNN', 'SVM', 'Random Forest','DecisionTreeClassifier','LogisticRegression','GradientBoostingClassifier', 'XGBClassifier')
    )

    # 3. Function for Datasets
    def get_dataset(name):
        data = None
        if name == 'Iris':
            data = datasets.load_iris()
        elif name == 'Wine':
            data = datasets.load_wine()
        elif name == 'Digits':
            data = datasets.load_digits()
        else:
            data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
        return X, y

    # 4. Dataset Overview
    st.subheader('Dataset Overview')

    X, y = get_dataset(dataset_name)
    st.write('Shape of dataset:', X.shape)
    st.write('Number of classes:', len(np.unique(y)))

    # 5. Function for HyperParameters
    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 10.0)
            params['C'] = C
            st.sidebar.warning(' C [default = 0.1]')
            kernel = st.sidebar.selectbox('kernel',('linear','poly', 'rbf','sigmoid'))
            params['kernel'] = kernel
            st.sidebar.warning('kernel [default = rbf]')
            degree = st.sidebar.slider('degree', 3, 20)
            params['degree'] = degree
            st.sidebar.warning("degree [default = 3 ")
            gamma = st.sidebar.selectbox('gamma',('auto','scale'))
            params['gamma'] = gamma
            st.sidebar.warning('gamma [default = auto]')
            decision_function_shape = st.sidebar.selectbox('decision_function_shape',('ovr', 'ovo'))
            params['decision_function_shape'] = decision_function_shape
            st.sidebar.warning('decision_function_shape [default = ovr]')
            random_state =   st.sidebar._number_input('random state',min_value=0)
            params['random state'] = random_state
            st.sidebar.warning('random state [default = None]')
        elif clf_name == 'KNN':
            n_neighbors = st.sidebar.slider('n_neighbors', 1, 15)
            params['n_neighbors'] = n_neighbors
            st.sidebar.warning('n_neighbors [default = 5]')
            weights = st.sidebar.selectbox('weights', ('uniform', 'distance'))
            params['weights'] = weights
            st.sidebar.warning('weights [default = uniform]')
            algorithm = st.sidebar.selectbox('algorithm',('auto','ball_tree', 'kd_tree', 'brute'))
            params['algorithm'] = algorithm
            st.sidebar.warning('algorithm [default = auto]')
            leaf_size = st.sidebar.slider('leaf_size',30,100)
            params['leaf_size'] = leaf_size
            st.sidebar.warning('leaf_size [default = 30]')
        elif clf_name == 'DecisionTreeClassifier':
            criterion = st.sidebar.selectbox('criterion',('gini', 'entropy', 'log_loss'))
            st.sidebar.warning('criterion [default = gini]')
            splitter = st.sidebar.selectbox('splitter',('best','random'))
            st.sidebar.warning('splitter [default = best]')
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            st.sidebar.warning('max depth [default = None]')
            params['criterion'] = criterion
            params['max_depth'] = max_depth
            params['splitter'] = splitter
        elif  clf_name == 'LogisticRegression':
            C = st.sidebar.slider('C', 0.01, 10.0)
            params['C'] = C
            st.sidebar.warning('C [default = 1.0]')
            penalty = st.sidebar.selectbox('penalty',('l2',  'none'))
            params['penalty'] = penalty
            st.sidebar.warning('penalty [default = l2]')
            solver = st.sidebar.selectbox('solver',('lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga' ))
            params['solver'] = solver
            st.sidebar.warning('solver [default = lbfgs]')
        elif clf_name == 'GradientBoostingClassifier':
            loss = st.sidebar.selectbox('loss', ('log_loss', 'deviance', 'exponential'))
            params['loss'] = loss
            st.sidebar.warning('loss [default = log_loss]')
            criterion = st.sidebar.selectbox('criterion', ('friedman_mse', 'squared_error', 'mse'))
            params['criterion'] = criterion
            st.sidebar.warning('criterion [default = friedman_mse]')
            learning_rate = st.sidebar.slider('learning_rate', 0.1, 1.0)
            params['learning_rate'] = learning_rate
            st.sidebar.warning('learning_rate [default = 0.1]')
            n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            params['n_estimators'] = n_estimators
            st.sidebar.warning('n_estimators [default = 100]')
            subsample = st.sidebar.slider('subsample', 0.1, 1.0)
            params['subsample'] =subsample
            st.sidebar.warning('subsample [default = 1.0]')
            min_weight_fraction_leaf = st.sidebar.slider('min_weight_fraction_leaf', 0.0, 0.5)
            params['min_weight_fraction_leaf'] = min_weight_fraction_leaf
            st.sidebar.warning('min_weight_fraction_leaf [default = 0.0]')
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
            st.sidebar.warning('max_depth [default = 3')
        elif clf_name == 'XGBClassifier':
            booster = st.sidebar.selectbox('booster',('gbtree', 'gblinear', 'dart'))
            params['booster'] = booster
            sampling_method = st.sidebar.selectbox('sampling_method',('uniform', 'gradient_based'))
            params['sampling_method'] = sampling_method
            eta = st.sidebar.slider('learning rate', 0.0, 1.0)
            params['learning rate'] = eta
            st.sidebar.warning('learning rate [deafult = 0.3]')
            max_depth = st.sidebar.slider('max_depth', 2, 15 )
            params['max_depth'] = max_depth
            st.sidebar.warning('max depth [default = 6]')
            gamma = st.sidebar.slider('min split loss', 0, 100)
            params['min split loss'] = gamma
            st.sidebar.warning('min split loss [default = 0] The larger gamma is, the more conservative the algorithm will be.')

        else:
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
            st.sidebar.warning('max depth [default = None]')
            n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            params['n_estimators'] = n_estimators
            st.sidebar.warning('n estimators [default = 100]')
            criterion = st.sidebar.selectbox('criterion', ('gini', 'entropy', 'log_loss'))
            params['criterion'] = criterion
            st.sidebar.warning('criterion [default = gini]')
            bootstrap = st.sidebar.selectbox('bootstrap',('True','False'))
            params['bootstrap'] = bootstrap
            st.sidebar.warning('bootstrap [default = True]')
            oob_score = st.sidebar.selectbox('oob_score',('False','True'))
            params['oob_score'] = oob_score
            st.sidebar.warning('oob score [default = False]')
            max_samples = st.sidebar.slider('max_samples', 0.1, 1.0)
            params['max_samples'] = max_samples
            st.sidebar.warning('max samples [default = None]')

        return params

    params = add_parameter_ui(classifier_name)

    # 6. Function for Classification
    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'SVM':
            clf = SVC(C=params['C'], kernel=params['kernel'], degree=params['degree'], gamma=params['gamma'],
                      decision_function_shape=params['decision_function_shape'], random_state=params['random state'])
        elif clf_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights=params['weights'],algorithm=params['algorithm'],
                                       leaf_size=params['leaf_size'])
        elif clf_name == 'DecisionTreeClassifier':
            clf = DecisionTreeClassifier(criterion=params['criterion'],max_depth=params['max_depth'],splitter=params['splitter'], random_state=1234)
        elif clf_name =="LogisticRegression":
            clf = LogisticRegression(C=params['C'], penalty=params['penalty'], solver=params['solver'])
        elif clf_name == 'GradientBoostingClassifier':
            clf = GradientBoostingClassifier(loss=params['loss'], criterion=params['criterion'], learning_rate=params['learning_rate'],
                                             n_estimators=params['n_estimators'],subsample=params['subsample'],
                                             min_weight_fraction_leaf=params['min_weight_fraction_leaf'],max_depth=params['max_depth'])
        elif clf_name == "XGBClassifier":
            clf = XGBClassifier(booster=params['booster'], sampling_method=params['sampling_method'],eta=params['learning rate'],
                                max_depth=params['max_depth'], gamma=params['min split loss'])
        else:
            clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],criterion=params['criterion'], bootstrap=params['bootstrap'],
                                         oob_score=params['oob_score'], max_samples=params['max_samples'], random_state=1234)
        return clf

    # 7. Model Fitting and predicting
    clf = get_classifier(classifier_name, params)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 8. Model Evaluation
    st.subheader('Model Evaluation Report')
    acc = (accuracy_score(y_test, y_pred))*100
    acc = str(round(acc,2))
    matrix = confusion_matrix(y_test, y_pred)
    f1_score = (f1_score(y_test, y_pred, average='macro'))*100
    f1_score = str(round(f1_score,2))
    precision_score = (precision_score(y_test, y_pred, average=None))*100
    recall_score = (recall_score(y_test, y_pred, average=None))*100

    st.write(f'Classifier Using = {classifier_name}')
    st.write(f'Accuracy =', acc, "%")
    st.write(f'F1 Score =',f1_score, '%')

    # Printing Table of Precision_score and Recall_score
    prec_score = []
    rec_score = []
    for p in precision_score:
        prec_score.append(float(p))
    for r in recall_score:
        rec_score.append(float(r))

    st.write(pd.DataFrame({
         'Precision Score': prec_score,
         'Recall Score': rec_score
     }))
    st.write(f'Confusion Matrix: ', matrix)

    st.subheader('Model Evaluation Report in Visualization')
    #### PLOT DATASET ####
    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    # Figure1. Accuracy and F1 Score Plot
    st.text('Accuracy and F1 Score Plot:')
    fig = plt.figure()
    x = ['F1 Score', 'Accuracy']
    h = np.array([float(f1_score), float(acc)])
    c = ['lightcoral', 'palegreen']
    plt.barh(x, h, 0.3, color = c)
    plt.xlabel('Score Value')
    for index, value in enumerate(h):
        plt.text(value, index,
                 str(value))
    st.pyplot(fig)

    # Figure2. Precision Score %
    st.text('Precision Score % :')
    fig = plt.figure()
    plt.bar(np.unique(y), precision_score,tick_label=np.unique(y) ,color=['silver','pink','grey','darksalmon','turquoise',
                                                  'plum','bisque','cornflowerblue','lightcoral','palegreen'])
    st.pyplot(fig)

    # Figure3. Recall Score %
    st.text('Recall Score % :')
    fig = plt.figure()
    plt.bar(np.unique(y), precision_score,tick_label=np.unique(y) ,color=['silver','pink','grey','darksalmon','turquoise',
                                                  'plum','bisque','cornflowerblue','lightcoral','palegreen'])
    st.pyplot(fig)

    # Figure4. Confusion Matrix
    st.text('Confusion Matrix:')
    fig =plt.figure()
    sns.heatmap(matrix, annot=True, fmt='d', cmap="YlGnBu")
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    st.pyplot(fig)

    # Figure5. Dataset Plot in 2D using PCA
    st.text('Dataset Plot in 2D using PCA:')
    fig = plt.figure()
    plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    st.pyplot(fig)

with tab2:
    st.info('This is Machine Learning algorithms hyperparameters tuning web app. In this i used \n'
            'pre-existing classification toy datasets from scikit-learn to analysis how various evaluation \n'
            'metrics (Accuracy score, F1 score, Precision score, Recall score, Confusion matrix) changes by \n '
            ' tuning hyperparameters of various machine learning classifiers (KNN, SVM, Random Forest, Decision Tree, \n'
            'Logisticregression, GradientBoosting, XGBoost). Its also having the default information of those hyperparameters \n'
            'such that if we use default classifier (without any hyperparameter tuning) then that classifier will be using \n'
            'those default values. By hyperparameter tuning we can get the most accurate hyperparameter values that will \n'
            'best fit to the model. '
             )
    st.text('Created by JAINARAYAN SINGH')
    st.text('References')
    st.write("scikit-learn documentation https://scikit-learn.org/stable/")
    st.write('xgboost documentation https://xgboost.readthedocs.io/en/stable/')
