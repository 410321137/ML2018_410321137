import os
import sys
import numpy as np
import pickle
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QDialog, QFileDialog
from sklearn.pipeline import Pipeline
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm
from mainwindow import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self) #初始化ui
        self.pushButton.clicked.connect(self.select_file)
        self.pushButton_2.clicked.connect(self.predict)

    def select_file(self):
        path = QFileDialog.getOpenFileName()
        self.lineEdit.setText(path[0])

    def predict(self):
        pipe = open('pipe1.pickle', 'rb')
        pipe = pickle.load(pipe)

        path = self.lineEdit.text()
        img = imread(path, mode='L')
        img = imresize(img, (30, 40))
        img = img.flatten()

        test = []
        test.append(img)
        predicted = pipe.predict(test)
        print(predicted)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    with open('style.qss', 'r') as filepath:
        app.setStyleSheet(filepath.read()) #設定window style
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())