# Assignment2
### Running Result
    
![](ss2.png)

### Report and Discussion
首先先將數據集讀入

    mnist = fetch_mldata('MNIST original')
    #拆成訓練跟測試用
    train_x, test_x, train_y, test_y = train_test_split( mnist.data, mnist.target,test_size=1 / 7.0, random_state = 0)

這裡用6萬筆訓練，1萬筆測試

根據建議的方式正規化

    #標準化
    train_x = train_x / 255
    test_x = test_x / 255

查看一下該下降的維度包含的特徵

    pca = PCA()
    pca.fit(train_x)
    for i in np.arange(0.5, 1.0, 0.05):
        n = find_d(pca.explained_variance_ratio_, i)
        print(i,n)

其結果為

    0.5 11
    0.55 14
    0.60 17
    0.650 21
    0.70 26
    0.75 33
    0.80 43
    0.85 59
    0.90 87
    0.95 154

可以看見該資料集在154的維度可以顯示95%的特徵，因為154維在6萬筆資料中將會執行好一段時間，所以這裡就先用16維稍微試試看

    pca = PCA(n_components = 16)
    train_x_reduced = pca.fit_transform(train_x)
    test_x_reduced = pca.transform(test_x)

維度下降完畢後開始使用辨識器看結果如何

    clf = svm.SVC(C = 0.1, kernel='rbf', gamma = 0.1)
    clf = clf.fit(train_x_reduced, train_y)
    predict = clf.predict(test_x_reduced)
    print("Classification report for SVM classifier: \n %s\n\n%s\n"
      % (clf, metrics.classification_report(test_y, predict)))

    clf2 = LogisticRegression(random_state= 1)
    clf2 = clf2.fit(train_x_reduced, train_y)
    predict = clf2.predict(test_x_reduced)
    print("Classification report: \n %s\n\n%s\n"
      % (clf2, metrics.classification_report(test_y, predict)))
    
這裡使用SVM與LR辨識

![](ss2.png)

結果是出人意料的好，由其SVM到達96%的準確率，LR因其線性關係稍弱，不過也算得到不錯的成績

### Result Discussion


### What I learn
