# Assignment3
### Running Result

![](ss1.png) ![](ss2.png)



### Report and Discussion
在 train.py 中

首先讓使用者讀入測試資料，產生測試集，並在這裡縮小一下圖片的大小

        img = imread(path + '/' + f , mode='L')
        img = imresize(img, (30, 40))

然後根據訓練的需求，產生由圖片跟標籤(正確答案)產生的array，train_x, train_y, test_x, test_y

為了方便之後調用，利用pickle的功能將array保存，就不用再次讀取並跑回圈

    f = open('train_x.pickle', 'wb')
        pickle.dump(train_x, f)
        f.close()

讀取完測試集資料後就開始標準化

    #標準化
        scaler = StandardScaler()
        self.train_x_stand = scaler.fit_transform(self.train_x)
        self.test_x_stand = scaler.fit_transform(self.test_x)

然後用PCA的方法降低維度

    pca = PCA(0.95)
        self.train_x_reduced = pca.fit_transform(self.train_x_stand)
        self.test_x_reduced = pca.transform(self.test_x_stand)
        self.save_reduce(self.train_x_reduced, self.test_x_reduced)

保留95%的方差百分比

最後找辨識器開始訓練，這裡使用三種，分別是LogisticRegression、SVM、GaussianNB，並且有結合三種辨識器的VotingClassifier

    clf1 = LogisticRegression(random_state=1)
    clf2 = SVC(C = 100, gamma = 0.0001, probability = True)
    clf3 = GaussianNB()
    eclf = VotingClassifier(estimators = [('lr', clf1), ('svm', clf2), ('gnb', clf3)], weights= [1.5, 3, 1.5], voting = 'hard')

在調整SVM參數的時候使用了sklearn的GridSearchCV進行了自動調整

    param_grid = {'C': [1, 10, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(train_x_reduced, train_y)
    print(clf.best_estimator_)

以下是各個辨識器對test data做辨識的時候得到的準確率

 * LogisticRegression : 0.85
 * SVM : 0.93
 * GaussianNB : 0.78
 * VotingClassifier : 0.91

在測試集中的準確率由sklearn提供的report函數可得知

    metrics.classification_report(self.test_y, predict))

可以發現線性辨識器在這個資料集中居然有相當不錯的成績，而SVM此種經典算法在這裡取得最好的成績，GaussianNB在這裡較差強人意一些，我是聽說該方法對不算大量的資料集會相當有用才使用的，從結果來看可能並不適合這種問題?
總之看過個別表現後可以確定在嘗試結合辨識器的時候，應該讓SVM的權重大一些

然而結合後的辨識器經過一些權重的調整後，仍然無法突破SVM的準確率，於是在這裡決定不使用結合的辨識器，只使用SVM可能會帶來更好的辨識結果

決定了辨識器並且訓練過後，將結果的模型做保存

    pipe2 = Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=100)),
                    ('clf', SVC(C = 100, gamma = 0.0001))
                    ])
    pipe2.fit(self.train_x, self.train_y)

    f = open('pipe1.pickle', 'wb')
    pickle.dump(pipe1, f)
    f.close()

在 main.py 中

首先讓使用者輸入照片，照片從檔案中打開

    path = QFileDialog.getOpenFileName()

之後將先前訓練好的模型讀入

    pipe = open('pipe2.pickle', 'rb')
    pipe = pickle.load(pipe)

讓模型做出預測

    predicted = pipe.predict(test)


### Result Discussion
![](ss3.png)
這是經過PCA降維後，對前兩個主成分畫圖做成的結果，因為PCA是無監督式的學習，所以標籤的顏色是根據本來的label標的，雖然只有前兩個主成分，而且此次的降維也只是降到100維罷了，並且前二主成分占的比重也就差不多16-17%左右吧，所以還是難以代表全部，在二維圖上看上去就擠成一坨，只是還是稍微看得出各個類別的特徵向量都沒跑偏太多，或許是因為這樣所以線性的分類器也能得出不錯的表現。

![](ss4.png)

這是降維後主成分的占比，由大到小排列

![](ss5.png)

這是當初找降下的維度時所找到的維度，可以看見若想保持有95%的特徵那麼降到100維度可能是比較正確的

然後理所當然的可以解非線性問題的SVM表現的較LR好，然而可能是因為資料集的特徵上的一些問題，GNB分類器表現的就沒那麼好。

我覺得機器學習這方面雖然寫起來並不是太難(畢竟大部分人家都實作好了)，但是後面的理論跟各式各樣的實作方法要去了解還是很花時間，當然只跟著範例打還是可以弄出點東西，回想當初有去參加過學校辦的機器學習的講座，因為時間有限的關係講者就直接讓我們import資料集然後開始找一些分類器訓練，結果三小時過去我感覺甚麼也不太清楚，連自己餵了什麼給模型都搞不太懂，那時似乎是用的鳶尾花資料集吧，總而言之這領域的東西確實不是一時半會就能搞懂或者說明白的，所以上課跟做這個作業的時候查到資料有時都有種原來如此的感覺，感覺倒也不壞。

另外我覺得雖然模型訓練的code是比較簡單，很多東西不用自己實現，但是調參數可以說是大坑，我大部分的時間可能都花在查參數跟調整參數上了，畢竟有時候調下去影響不小，但一個一個的分類器中的參數背後又是一條一條的理論，要查跟測試可真是比較麻煩，然後像SVM這種運行比較慢的模型就感覺調的時候特別久了，上個作業就花了好久時間調參數，大量數據+SVM的執行時間，確實是相當的頭痛。最後我以前是認為機器學習都要餵入大量數據才會準確，但這次的人臉也就650張左右的資料集，在SVM下也可以達到9成以上的辨識準確率，稍微的刷新的一下觀念，確實利用提取特徵的話就可以在小量的資料集達成很好的分類，我覺得這點確實是相當優秀的想法。

### What I learn

批量讀取檔案與保存變數

使用機器學習模型做特徵的辨識

降低維度運行跟確認

模型參數調整
