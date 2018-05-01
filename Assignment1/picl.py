import cv2
import numpy as np
import sklearn
import random

def GDmethod(eprime, E, I, k1, k2):
    epoch = 1
    a = 0.00001
    MaxIterLimit = 10
    wt = list()
    ak = list()
    ek = list()
    while(epoch ==1 or epoch < MaxIterLimit):
        for i in range (0, eprime.shape[0]):
            for j in range (0, emprime.shape[1]):
                wt.append([random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)])
                xk =  k1[i, j] + wt[i][1] * k2[i][j] + wt[i][2] * k2[i][j]
                ek = [E[i][j] - ak[i][0], E[i][j] - ak[i][1], E[i][j] - ak[i][2]]

    def res(w, x):
        return w.T.dot(x)


if __name__ == "__main__":

    E_prime = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData/Eprime.png')
    E = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\E.png')
    I = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\I.png')
    key1 = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\key1.png')
    key2 = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\key2.png')

    GDmethod(eprime, E, I, key1, key2)

    '''
    f_key1 = open('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\k1.txt', 'r')
    f_key2 = open('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\k2.txt', 'r')
    f_eprime = open('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\eprime.txt', 'r')
    f_i = open('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\i.txt', 'r')
    cv2.imshow('image', E_prime)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''