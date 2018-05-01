import cv2
import numpy as np
import random
import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QDialog, QGraphicsView
from mainwindow import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self) 
        self.eprime = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData/Eprime.png', cv2.IMREAD_GRAYSCALE)
        self.E = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\E.png', cv2.IMREAD_GRAYSCALE)
        self.I = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\I.png', cv2.IMREAD_GRAYSCALE)
        self.key1 = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\key1.png', cv2.IMREAD_GRAYSCALE)
        self.key2 = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\key2.png', cv2.IMREAD_GRAYSCALE)
        self.pushButton.clicked.connect(self.GDmethod)

    def GDmethod(self):
        epoch = 1
        a = 0.00001
        MaxIterLimit = int(self.lineEdit.text())
        wt = np.array([random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)], np.int32)
        ak = list()
        ek = list()
        we = np.array([1, 1, 1])
        ee = float(self.lineEdit.text())
        IE = self.eprime.copy()
        IE2 = self.E.copy()
        while(epoch == 1 or epoch < MaxIterLimit):
            for i in range (0, self.eprime.shape[0]):
                for j in range (0, self.eprime.shape[1]):
                    xk = np.array([self.key1[i][j], self.key2[i][j], self.I[i][j]])
                    ak = self.res(wt, xk)
                    ek = self.E[i][j] - ak
                    sk = a * ek * xk
                    temp = wt.copy()
                    wt = wt + sk
            we = np.array([wt[0] / we[0], wt[1] / we[1], wt[2] / we[2]])
            if(we[0] < 1 + ee and we[0] > 1 - ee and we[1] < 1 + ee and we[1] > 1 - ee and we[1] < 1 + ee and we[1] > 1 - ee):
                break
            epoch = epoch + 1
        self.label_3.setText("訓練次數: " + str(epoch))

        for i in range (0, self.eprime.shape[0]):
                for j in range (0, self.eprime.shape[1]):
                    IE[i][j] = ( self.eprime[i][j] - wt[0] * self.key1[i][j] - wt[1] * self.key2[i][j] ) / wt[2]

        for i in range (0, self.eprime.shape[0]):
                for j in range (0, self.eprime.shape[1]):
                    IE2[i][j] = ( self.E[i][j] - wt[0] * self.key1[i][j] - wt[1] * self.key2[i][j] ) / wt[2]

        cv2.imshow('image', IE)
        cv2.imshow('image2', IE2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def res(self, w, x):
        a = w.T.dot(x)
        return a

if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    # python .\main.py
    # pyuic5 mainwindow.ui -o mainwindow.py