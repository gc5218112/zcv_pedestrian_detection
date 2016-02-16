pedestrian_HOG_SVM_Inria_Caltech工程简介
===
by baiyu33

## 功能
基于OpenCV在Inria行人检测数据集上，做训练和测试，生成bbs用于评测

## 代码
在OpenCV3.0中的例程opencv/sources/sample/cpp/train_HOG.cpp基础上稍作修改
即：HOG+SVM实现做行人检测的训练和测试

## 数据
使用Caltech转化过的Inria行人检测数据集进行训练和测试
可通过下载如下item生成转化过的数据：
1.Caltech主页的Piotr Dollar提供的matlab toolbox
2.Caltech主页的用于转换图片和标注信息的代码
3.Caltech主页的转化过的Inria数据集
训练正样本：根据bbox从positive train images中截取，并resize到64*128的大小
训练负样本：从没有人的negative train images中随机取，每张图取10张：顶点随机生成，而宽度与高度是固定的64*128
训练的hard example：先用正样本和负样本训练得到模型，用这个模型在negative training images上（也就是没有人的背景图上）检测行人，检测到的都是false positive example。这些样本作为增加的训练负样本，与训练正样本共同参与到第二次训练

第二次训练完毕后，用得到的模型在测试机上检测，每次detectMultiScale的时候都能都到如下bbs信息:
imageId, x, y, width, height, score
即：图片id，检测窗口的左上角x坐标，y坐标，窗口宽度，高度，评分
所有这样的bbs信息都写入文件，就可以交给Piotr Dollar的那个matlab toolbox做评测了，稍微修改代码后执行dbEval就得到miss rate - fppi的结果。

不过结果真的不太理想，miss rate高达72%。换用OpenCV的hog默认的行人检测模型参数，在测试集上做检测并生成bbs文件，评测miss rate仍然高达71%。而Caltech官方评测中HOG的miss rate是46%，看来同志仍需努力啊。

## 代码下载
到csdn搜用户baiyu33的资源