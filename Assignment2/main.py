import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

def show_image_d(train_img, train_lbl, img_name):
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
        new_img_7.paste(im, (x_offset,0)) #黏貼圖片
        x_offset += im.size[0]
    new_img_7.save(img_name) #保存圖片

if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original')
    #將數據拆成訓練跟測試用
    train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1 / 7.0, random_state = 0)

    #查看原始圖長相
    show_image(train_img, train_lbl, 'test1.jpg')

    #標準化
    train_img = train_img / 255
    test_img = test_img / 255

    show_image(train_img * 255 , train_lbl, 'test3.jpg')

    #PCA降維
    '''
    for i in np.arange(0.95, 0.97, 0.01):
        pca = PCA(i)
        pca.fit(train_img)
        print(pca.n_components_)
    '''

    pca = PCA(n_components=10)
    train_img_reduced = pca.fit_transform(train_img)
    train_img_recovered = pca.inverse_transform(train_img_reduced)
    show_image(train_img_recovered * 255 , train_lbl, 'test4.jpg')

    pca = PCA(n_components=30)
    train_img_reduced = pca.fit_transform(train_img)
    train_img_recovered = pca.inverse_transform(train_img_reduced)
    show_image(train_img_recovered * 255 , train_lbl, 'test5.jpg')

    pca = PCA(n_components=50)
    train_img_reduced = pca.fit_transform(train_img)
    train_img_recovered = pca.inverse_transform(train_img_reduced)
    show_image(train_img_recovered * 255 , train_lbl, 'test6.jpg')

    pca = PCA(n_components=154)
    train_img_reduced = pca.fit_transform(train_img)
    train_img_recovered = pca.inverse_transform(train_img_reduced)
    show_image(train_img_recovered * 255 , train_lbl, 'test7.jpg')




