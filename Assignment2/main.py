import numpy as np
import matplotlib.pyplot as plt
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

def show_image(train_img, train_lbl, img_name):
    img_7 = []
    for i in range(1000):
      if train_lbl[i] == 7 and len(img_7) < 10:
          img_7.append(train_img[i])
    new_img_7 = Image.new('L', (280, 28)) #開10張圖片的大小

    x_offset = 0
    for im in img_7:
        im = np.reshape(im, (28,28))
        #print(im.shape)
        im = Image.fromarray(np.uint8(im), 'L') #array轉換成圖片格式
        #im.show()
        new_img_7.paste(im, (x_offset,0)) #黏貼圖片
        x_offset += im.size[0]
    new_img_7.save(img_name) #保存圖片

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
    mnist = fetch_mldata('MNIST original')
    #將數據拆成訓練跟測試用
    train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1 / 7.0, random_state = 0)
    #查看原始圖長相
    show_image(train_img, train_lbl, 'test1.jpg')

    #標準化
    train_img = train_img / 255
    test_img = test_img / 255

    #show_image(train_img * 255 , train_lbl, 'test3.jpg')

    #PCA降維
    '''
    pca = PCA()
    pca.fit(train_img)
    for i in np.arange(0.5, 1.0, 0.1):
        n = find_d(pca.explained_variance_ratio_, i)
        print(i,n)
    '''
    #從上面得到的結果先進行試驗
    pca = PCA(n_components = 49)
    train_img_reduced = pca.fit_transform(train_img)
    test_img_reduced = pca.fit_transform(test_img)
    #print(train_img_reduced.shape, train_lbl.shape)

    '''
    #尋找SVM的參數
    param_grid = { "C" : [0.1], "gamma" : [0.1]}
    rf = svm.SVC()
    gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=2, n_jobs=-1, verbose=1)
    gs = gs.fit(train_img_reduced, train_lbl)
    print(gs.best_score_)
    print(gs.best_params_)
    '''

    #bp = gs.best_params_
    #clf = svm.SVC(C=bp['C'], kernel='rbf', gamma=bp['gamma'])
    clf = svm.SVC(C = 0.1, kernel='rbf', gamma = 0.1)
    clf = clf.fit(train_img_reduced, train_lbl)
    predict = clf.predict(test_img_reduced)
    print("Classification report for SVM classifier: \n %s\n\n%s\n"
      % (clf, metrics.classification_report(test_lbl, predict)))
    #print(svc,svc.score(test_img_reduced, test_lbl))
    #train_img_recovered = pca.inverse_transform(train_img_reduced)
    #show_image(train_img_recovered * 255 , train_lbl, 'test7.jpg')

    #python .\main.py

    






