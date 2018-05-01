import cv2
import numpy as np
import random

def GDmethod(eprime, E, I, k1, k2):
    epoch = 1
    a = 0.00001
    MaxIterLimit = 5
    wt = np.array([random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)], np.int32)
    ak = list()
    ek = list()
    IE = eprime.copy()
    while(epoch ==1 or epoch < MaxIterLimit):
        for i in range (0, eprime.shape[0]):
            for j in range (0, eprime.shape[1]):
                xk = np.array([k1[i][j], k2[i][j], I[i][j]])
                ak = res(wt, xk)
                ek = E[i][j] - ak
                sk = a * ek * xk
                wt = wt + sk
        
        epoch = epoch + 1
        print(epoch)

    for i in range (0, eprime.shape[0]):
            for j in range (0, eprime.shape[1]):
                IE[i][j] = ( eprime[i][j] - wt[0] * k1[i][j] - wt[1] * k2[i][j] ) / wt[2]
    
    cv2.imshow('image', IE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def res(w, x):
    a = w.T.dot(x)
    return a


if __name__ == "__main__":

    eprime = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData/Eprime.png', cv2.IMREAD_GRAYSCALE)
    E = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\E.png', cv2.IMREAD_GRAYSCALE)
    I = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\I.png', cv2.IMREAD_GRAYSCALE)
    key1 = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\key1.png', cv2.IMREAD_GRAYSCALE)
    key2 = cv2.imread('C:/Users/pistori/Documents/Github/IOML2018spring/Assignment1/ImageData\key2.png', cv2.IMREAD_GRAYSCALE)

    GDmethod(eprime, E, I, key1, key2)