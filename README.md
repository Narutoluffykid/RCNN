# 数据集
采用17flowers据集, 官网下载：http://www.robots.ox.ac.uk/~vgg/data/flowers/17/

# 论文思想
RCNN主要分为三个步骤：第一使用selective search在图片中选取2000个region proposal。第二对每一个region proposal用卷积神经网络（5层卷积+2层全连接）提取特征向量。第三使用一组线性SVM对特征向量进行分类。

# felzenswalb(图像分割)：
Step1:计算每个像素点与其8领域或4领域的不相似度/n
Step2:将边按照不相似度从小到大排序，e1,e2 …… en
Step3:选择e1
Step4:对当前所选择的边en进行合并判断，设en连接的点为{Vi,Vj},如果其满足合并条件
（1）	两个顶点不属于同一区域{Id(vi)≠Id(vj)}
（2）	两个顶点的不相似度大于两者内部的不相似度{wij>Mint(ci,cj)}
合并并且更新
（1）	将Id(vi),Id(vj)的类别标号统一为Id(vi)
（2）	将类内不相似度更新为wi+k/(|ci|+|cj|)
Step5:如果n<=N,按照排好的顺序选择下一条边进行Step4

# Selective search：
Step1:使用图像分割技术生成初始化区域集R
Step2:计算区域集R里面每一个相邻区域的相似度S={s1、s2......}
Step3:把相似度最高的两个区域合并为一个区域，添加进R
Step4:删除step2中用于合并成step3区域的子区域
Step5:计算新区域集的相似度
Step6:跳至step2直到s1、s2......为空

# Alexnet:
1.	network = input_data(shape=[None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
2.	    network = conv_2d(network, 96, 11, strides=4, activation='relu')  
3.	    network = max_pool_2d(network, 3, strides=2)  
4.	    network = local_response_normalization(network)     #局部响应归一化（batch_normalization的出现几乎替代了local_response_normalization）  
5.	    network = conv_2d(network, 256, 5, activation='relu')  
6.	    network = max_pool_2d(network, 3, strides=2)  
7.	    network = local_response_normalization(network)  
8.	    network = conv_2d(network, 384, 3, activation='relu')  
9.	    network = conv_2d(network, 384, 3, activation='relu')  
10.	    network = conv_2d(network, 256, 3, activation='relu')  
11.	    network = max_pool_2d(network, 3, strides=2)  
12.	    network = local_response_normalization(network)  
13.	    network = fully_connected(network, 4096, activation='tanh')  
14.	    network = dropout(network, 0.5)  
15.	    network = fully_connected(network, 4096, activation='tanh')  
16.	    network = dropout(network, 0.5)  
17.	    network = fully_connected(network, num_classes, activation='softmax')  
18.	    network = regression(network, optimizer='momentum',  
19.	                         loss='categorical_crossentropy',  
20.	                         learning_rate=0.001)  
21.	    return network  
Fine_tune_Alexnet:去掉最后一个全连接层

非极大值抑制：真实标记框的位置与selective search算法得出的region proposal做IOU，IOU小于设定阈值将region proposal修改标签为0（背景）。

# 文件说明
config.py:路径配置文件，存放alexnet和svm模型的保存路径等信息
train_alexnet.py：训练alexnet
fine_tune_RCNN.py:对Alexnet网络进行微调
preprocessing_RCNN.py:图像处理（shape_resize\IOU\用ndarray格式存储region proposal）
RCNN_output.py:训练svm，预测结果
selectivesearch.py:选择性搜索
setup.py:创建文件夹，配合config使用
tools.py：画rectangle和进度条


# 参考
https://github.com/yangxue0827/RCNN （RCNN论文复现）
https://blog.csdn.net/m_z_g_y/article/details/81281398 （selective search的实现）
https://blog.csdn.net/Tomxiaodai/article/details/81412354 （selective search代码讲解）
