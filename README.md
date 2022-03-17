# 使用多层感知机解决多分类问题

1120191286			周彦翔

[toc]

## 实验目的

​	参考多层感知机建模和训练方法，利用交叉熵、梯度下降训练模型，实现给定样本分类。

​	训练数据为150个带噪声的标记数据。

## 多层感知机模型

​	使用的多层感知机的结构为：

+ 输入层，4个神经元
+ 隐层，7个神经元
+ 输出层，3个神经元

​	其中隐层和输出层均为全连接层，激活函数均为ReLU函数，输出层的输出最后经过Softmax函数进行归一化。

## 损失函数

​	损失函数采用交叉熵，由于真实结果是one-hot形式的，所以交叉熵可以简化为$$-\log{output[j]}$$，j为真实类别的编号。

## 优化方法

​	预先编写随机梯度下降、小批量梯度下降、动量法、自适应梯度法、Adam法，届时比较各个方法之间的优劣。

## 反向传播计算梯度

![计算图](C:\CODE\Python\Intelligent_computation_homework2\README.assets\计算图-16473441456321.svg)

​	根据计算图计算反向传播后交叉熵损失对各个分量的导数，如上图所示。