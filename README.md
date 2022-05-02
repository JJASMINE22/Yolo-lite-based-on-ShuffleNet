## 基于ShuffleNet骨干的YOLO目标检测模型 --tensorflow2
---

## 目录  
1. [所需环境 Environment](#所需环境) 
2. [注意事项 Attention](#注意事项)
3. [单元结构 Unit Structure](#单元结构)
4. [网络结构 Network Structure](#网络结构)
5. [数据下载 Download](#数据下载)
6. [模型文件 Model Files](#模型文件)
7. [训练步骤 Train](#训练步骤) 

## 所需环境  
1. Python3.7
2. Tensorflow-gpu>=2.4.0  
3. Numpy==1.19.5
4. Pillow==8.2.0
5. Opencv-contrib-python==4.5.1.48
6. CUDA 11.0+
7. Cudnn 8.0.4+

## 注意事项  
1. 若tensorflow版本高于2.5，cudnn版本需高于8.1.0
2. 将传统YOLO骨干替换为ShuffleNet，加入组卷积膨胀系数
3. 更新传统YOLO的softmax、sigmoid输出激活方式，解决多类目标识别效果差的问题
4. 更新基于携带多分类目标掩码的NLL误差，并解决tensorflow在自定义误差的条件下不便于贪婪执行的弊端
5. 更新推理时检测框体的检出标准
6. 加入正则化操作，降低过拟合影响
7. 数据与标签路径、训练参数等均位于config.py  
注：该项目缺少部分关键代码，诸如误差计算、目标检出标准等。关注并联系作者，说明用意以获取，email:m13541280433@163.com  

## 单元结构  
1. ShuffleUnit残差结构  
![image](图片地址)  
2. ShuffleUnit拼接结构  
![image](图片地址)  

## 网络结构
YOLO based on ShuffleNet  
![image](图片地址)  

## 数据下载    
coco2017  
链接：https://cocodataset.org/#home  
下载解压后将数据集放置于config.py中指定的路径。 

## 模型文件
链接：https://pan.baidu.com/s/16DpaXkJUnpKI4M1BsQp-UQ  
提取码：yolo  
下载后可分别置换项目中.\\tf_model\\checkpoint与.\\model_data目录下的内容  

## 训练步骤
运行train.py

## 预测步骤
运行predict.py
