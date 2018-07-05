import os
import sys
import numpy as np
import pickle
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont, QImage
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
        self.setupUi(self) 
        self.pushButton.clicked.connect(self.select_file)
        self.pushButton_2.clicked.connect(self.predict)

    def select_file(self):
        path = QFileDialog.getOpenFileName()
        self.lineEdit.setText(path[0])

    def predict(self):
        pipe = open('pipe2.pickle', 'rb')
        pipe = pickle.load(pipe)
        
        pipe2 = open('pipe3.pickle', 'rb')
        pipe2 = pickle.load(pipe2)
        
        pipe3 = open('pipe4.pickle', 'rb')
        pipe3 = pickle.load(pipe3)

        path = self.lineEdit.text()
        img = imread(path, mode='L')
        img_s = Image.fromarray(img, 'L')
        img = imresize(img, (30, 40))
        img = img.flatten()

        test = []
        test.append(img)
        predicted = pipe.predict(test)
        self.show_result(img_s, predicted)
        predicted2 = pipe2.predict(test)
        predicted3 = pipe3.predict(test)
        self.label_6.setText(' ')
        if(predicted2[0] != predicted[0] or predicted3[0] != predicted[0]):
            if(predicted2[0] == predicted3[0] and predicted2[0] != predicted[0]):
                self.label_6.setText('也有可能是: 第 ' + str(predicted2[0]) + ' 人')
            elif(predicted2[0] != predicted[0] and predicted3[0] != predicted[0]):
                self.label_6.setText('也有可能是: 第 ' + str(predicted2[0]) + ' 人或第 ' + str(predicted3[0]) + ' 人')
            elif(predicted2[0] == predicted[0] and predicted3[0] != predicted[0]):
                self.label_6.setText('也有可能是: 第 ' + str(predicted3[0]) + ' 人')
            elif(predicted2[0] != predicted[0] and predicted3[0] == predicted[0]):
                self.label_6.setText('也有可能是: 第 ' + str(predicted2[0]) + ' 人')

            
    def show_result(self, img_s, predict):
        facelist = open('facelist.pickle', 'rb')
        facelist = pickle.load(facelist)

        img_s = img_s.resize((self.graphicsView.width() -2, self.graphicsView.height() -2))
        qim = ImageQt(img_s)
        pix = QtGui.QPixmap.fromImage(qim) 
        graphicscene = QtWidgets.QGraphicsScene() 
        graphicscene.addPixmap(pix)
        self.graphicsView.setScene(graphicscene)
        self.graphicsView.show()

        print(predict)
        pred_face = facelist[predict[0] - 1]
        pred_face = pred_face.resize((self.graphicsView.width() -2, self.graphicsView.height() -2))
        qim = ImageQt(pred_face)
        pix = QtGui.QPixmap.fromImage(qim) 
        graphicscene = QtWidgets.QGraphicsScene() 
        graphicscene.addPixmap(pix)
        self.graphicsView_2.setScene(graphicscene)
        self.graphicsView_2.show()

        self.label_5.setText('第  ' + str(predict[0]) + '  人')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    with open('style.qss', 'r') as filepath:
        app.setStyleSheet(filepath.read())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    #python .\main.py
    #pyuic5 mainwindow.ui -o mainwindow.py