import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from PIL import Image
from time import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QDialog, QFileDialog
from sklearn.pipeline import Pipeline
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import fetch_mldata
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import metrics
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from mainwindow2 import Ui_MainWindow
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.data_gen)
        self.pushButton_2.clicked.connect(self.std_and_dr)
        self.pushButton_3.clicked.connect(self.train_SVM)
        self.pushButton_4.clicked.connect(self.train_clr)
        self.pushButton_5.clicked.connect(self.save_model)
        self.pushButton_6.clicked.connect(self.get_dir)

    def get_dir(self):
        self.lineEdit.setText(str(QFileDialog.getExistingDirectory(self, 'select directory')))

    def save_pic(self):
        path = 'C:/Users/pistori/Documents/Github/ML2018_410321137/Assignment3/Face'
        files = os.listdir(path)

        n = 0
        facelist = []
        for f in files:
            if not os.path.isdir(f):
                if(n % 13 == 0):
                    img = imread(path + '/' + f , mode='L')
                    img = Image.fromarray(img, 'L')
                    #imgI.show()
                    facelist.append(img)
                n = n + 1

        f = open('facelist.pickle', 'wb')
        pickle.dump(facelist, f)
        f.close()

    def data_gen(self):
        self.save_pic()
        #產生測試集資料
        train_x, train_y, test_x, test_y = self.load_data()
        #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        self.savefile(train_x, train_y, test_x, test_y)
        self.label_2.setText('測試集產生完畢')

    def load_data(self):
        #path = 'C:/Users/pistori/Documents/Github/ML2018_410321137/Assignment3/Face'
        path = self.lineEdit.text()
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

    def savefile(self, train_x, train_y, test_x, test_y):
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

    def open_trainset(self):
        train_x = open('train_x.pickle', 'rb')
        train_x = pickle.load(train_x)
        train_y = open('train_y.pickle', 'rb')
        train_y = pickle.load(train_y)
        test_x = open('test_x.pickle', 'rb')
        test_x = pickle.load(test_x)
        test_y = open('test_y.pickle', 'rb')
        test_y = pickle.load(test_y)

        return train_x, train_y, test_x, test_y
    
    def std_and_dr(self):
        #讀取
        self.train_x, self.train_y, self.test_x, self.test_y = self.open_trainset()
        #print(self.train_x.shape, self.train_y.shape, self.test_x.shape, self.test_y.shape)

        #標準化
        scaler = StandardScaler()
        self.train_x_stand = scaler.fit_transform(self.train_x)
        self.test_x_stand = scaler.fit_transform(self.test_x)

        #pca降維
        '''
        pca = PCA()
        pca.fit(train_x)
        for i in np.arange(0.5, 1.0, 0.05):
            n = self.find_d(pca.explained_variance_ratio_, i)
            print(i,n)
        '''
        #得到0.95的特徵比例集中在100個特徵內
        pca = PCA(0.95)
        self.train_x_reduced = pca.fit_transform(self.train_x_stand)
        self.test_x_reduced = pca.transform(self.test_x_stand)
        self.save_reduce(self.train_x_reduced, self.test_x_reduced)
        '''
        plt.scatter(self.train_x_reduced[:, 0], self.train_x_reduced[:, 1], c= self.train_y, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('jet', 50))
        plt.colorbar()
        plt.show()
        '''
        #print(self.train_x_reduced.shape, self.train_y.shape, self.test_x_reduced.shape, self.test_y.shape)
        
        #尋找合適參數
        #para = self.find_parameter(train_x_reduced, train_y)

        self.label_2.setText('標準化與降維完畢')

    def save_reduce(self, train_x_reduced, test_x_reduced):
        f = open('train_x_reduced.pickle', 'wb')
        pickle.dump(train_x_reduced, f)
        f.close()
        f = open('test_x_reduced.pickle', 'wb')
        pickle.dump(test_x_reduced, f)
        f.close()

    def train_LR(self):
        clf = LogisticRegression(random_state=1)
        t0 = time()
        clf.fit(self.train_x_reduced, self.train_y)
        predict = clf.predict(self.test_x_reduced)
        self.label_2.setText('花費時間: ' + str(time() - t0))
        self.textBrowser.setText(str(clf) + '\n' + str(metrics.classification_report(self.test_y, predict)))
    
    def train_SVM(self):
        clf2 = SVC(C = 100, gamma = 0.0001)
        t0 = time()
        clf2.fit(self.train_x_reduced, self.train_y)
        predict = clf2.predict(self.test_x_reduced)
        self.label_2.setText('花費時間: ' + str(time() - t0))
        self.textBrowser.setText(str(clf2) + '\n' + str(metrics.classification_report(self.test_y, predict)))
    
    def train_clr(self):
        clf1 = LogisticRegression(random_state=1)
        clf2 = SVC(C = 100, gamma = 0.0001, probability = True)
        clf3 = GaussianNB()

        eclf = VotingClassifier(estimators = [('lr', clf1), ('svm', clf2), ('gnb', clf3)], weights= [1.5, 3, 1.5], voting = 'hard')
        t0 = time()
        eclf.fit(self.train_x_reduced, self.train_y)
        predict = eclf.predict(self.test_x_reduced)
        self.label_2.setText('花費時間: ' + str(time() - t0))
        self.textBrowser.setText(str(eclf) + '\n' + str(metrics.classification_report(self.test_y, predict)))

    def save_model(self):
         #把模型封裝起來以便後續的調用
        clf1 = LogisticRegression(random_state=1)
        clf2 = SVC(C = 100, gamma = 0.0001, probability = True)
        clf3 = GaussianNB()
        eclf = VotingClassifier(estimators = [('lr', clf1), ('svm', clf2), ('gnb', clf3)], weights= [1.5, 3, 1.5], voting = 'hard')

        pipe1 = Pipeline([('sc', StandardScaler()),
                        ('pca', PCA(n_components=100)),
                        ('clf', eclf)
                        ])
        pipe1.fit(self.train_x, self.train_y)
        print('Test accuracy: %.3f' % pipe1.score(self.test_x, self.test_y))

        pipe2 = Pipeline([('sc', StandardScaler()),
                        ('pca', PCA(n_components=100)),
                        ('clf', SVC(C = 100, gamma = 0.0001))
                        ])
        pipe2.fit(self.train_x, self.train_y)
        print('Test accuracy: %.3f' % pipe2.score(self.test_x, self.test_y))

        pipe3 = Pipeline([('sc', StandardScaler()),
                        ('pca', PCA(n_components=100)),
                        ('clf', LogisticRegression(random_state= 1))
                        ])
        pipe3.fit(self.train_x, self.train_y)
        print('Test accuracy: %.3f' % pipe3.score(self.test_x, self.test_y))

        pipe4 = Pipeline([('sc', StandardScaler()),
                        ('pca', PCA(n_components=100)),
                        ('clf', GaussianNB())
                        ])
        pipe4.fit(self.train_x, self.train_y)
        print('Test accuracy: %.3f' % pipe4.score(self.test_x, self.test_y))

        f = open('pipe1.pickle', 'wb')
        pickle.dump(pipe1, f)
        f.close()
        
        f = open('pipe2.pickle', 'wb')
        pickle.dump(pipe2, f)
        f.close()

        f = open('pipe3.pickle', 'wb')
        pickle.dump(pipe3, f)
        f.close()

        f = open('pipe4.pickle', 'wb')
        pickle.dump(pipe4, f)
        f.close()

        self.label_2.setText('保存完畢')

    def find_d(self, explained_variance_ratio_, p):
        sum = 0
        n = 0
        for i in explained_variance_ratio_:
            sum = i + sum
            n = n + 1
            if(sum > p):
                break
        return n

    def find_parameter(self, train_x_reduced, train_y):
        param_grid = {'C': [1, 10, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        clf = clf.fit(train_x_reduced, train_y)
        print(clf.best_estimator_)

        return clf.best_params_

if __name__ == '__main__':
    app = QApplication(sys.argv)
    with open('style.qss', 'r') as filepath:
        app.setStyleSheet(filepath.read())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
 
    #python .\train.py
    #pyuic5 mainwindow2.ui -o mainwindow2.py




