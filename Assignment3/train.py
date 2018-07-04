import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from time import time
from sklearn.pipeline import Pipeline
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm

def load_data():
    path = 'C:/Users/pistori/Documents/Github/ML2018_410321137/Assignment3/Face'
    files = os.listdir(path)

    photo_n = 1
    count = 0
    people_n = 1
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for f in files:
        if not os.path.isdir(f):
            img = imread(path + '/' + f , mode='L')
            img = imresize(img, (30, 40))
            #print(img.shape)
            img = img.flatten()
            #print(img.shape)
            if(photo_n < 11):
                train_x.append(np.array(img))
                train_y.append(people_n)
                photo_n = photo_n + 1
            else:
                test_x.append(np.array(img))
                test_y.append(people_n)
                if(photo_n == 13):
                    photo_n = 0
                    people_n = people_n + 1
                photo_n = photo_n + 1
    
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def savefile(train_x, train_y, test_x, test_y):
    f = open('train_x.pickle', 'wb')
    pickle.dump(train_x, f)
    f.close()
    f = open('train_y.pickle', 'wb')
    pickle.dump(train_y, f)
    f.close()
    f = open('test_x.pickle', 'wb')
    pickle.dump(test_x, f)
    f.close()
    f = open('test_y.pickle', 'wb')
    pickle.dump(test_y, f)
    f.close()

def open_trainset():
    train_x = open('train_x.pickle', 'rb')
    train_x = pickle.load(train_x)
    train_y = open('train_y.pickle', 'rb')
    train_y = pickle.load(train_y)
    test_x = open('test_x.pickle', 'rb')
    test_x = pickle.load(test_x)
    test_y = open('test_y.pickle', 'rb')
    test_y = pickle.load(test_y)

    return train_x, train_y, test_x, test_y

def save_reduce(train_x_reduced, test_x_reduced):
    f = open('train_x_reduced.pickle', 'wb')
    pickle.dump(train_x_reduced, f)
    f.close()
    f = open('test_x_reduced.pickle', 'wb')
    pickle.dump(test_x_reduced, f)
    f.close()

def find_d(explained_variance_ratio_, p):
    sum = 0
    n = 0
    for i in explained_variance_ratio_:
        sum = i + sum
        n = n + 1
        if(sum > p):
            break
    return n

def find_parameter(train_x_reduced, train_y):
    param_grid = {'C': [1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(train_x_reduced, train_y)
    print(clf.best_estimator_)

    return clf.best_params_

if __name__ == '__main__':
    #產生測試集資料
    #train_x, train_y, test_x, test_y = load_data()
    #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    #savefile(train_x, train_y, test_x, test_y)

    #讀取
    train_x, train_y, test_x, test_y = open_trainset()
    #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    #pca降維
    '''
    pca = PCA()
    pca.fit(train_x)
    for i in np.arange(0.5, 1.0, 0.05):
        n = find_d(pca.explained_variance_ratio_, i)
        print(i,n)
    '''
    #得到0.95的特徵比例集中在100個特徵內
    #使用n_components = 100
    pca = PCA(n_components=100)
    train_x_reduced = pca.fit_transform(train_x)
    test_x_reduced = pca.fit_transform(test_x)
    #save_reduce(train_x_reduced, test_x_reduced)
    #print(train_x_reduced.shape, train_y.shape, test_x_reduced.shape, test_y.shape)

    #尋找合適參數
    para = find_parameter(train_x_reduced, train_y)

    pipe1 = Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=100)),
                    ('clf', LogisticRegression(random_state=1))
                    ])
    for i in range(10):
        pipe1.fit(train_x, train_y)
        print('Test accuracy: %.3f' % pipe1.score(test_x, test_y))
    #f = open('pipe1.pickle', 'wb')
    #pickle.dump(pipe1, f)
    #f.close()

    #python .\main.py





