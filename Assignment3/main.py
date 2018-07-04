import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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
            img = Image.open(path + '/' + f).convert('L')
            if(photo_n < 13):
                train_x.append(np.array(img))
                train_y.append(people_n)
                photo_n = photo_n + 1
            else:
                test_x.append(np.array(img))
                test_y.append(people_n)
                if(photo_n == 15):
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

def find_d(explained_variance_ratio_, p):
    sum = 0
    n = 0
    for i in explained_variance_ratio_:
        sum = i + sum
        n = n + 1
        if(sum > p):
            break
    return n

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data()
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    savefile(train_x, train_y, test_x, test_y)

    #python .\main.py





