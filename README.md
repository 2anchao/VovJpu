# VovJpu
## Model 
>> I use the vovnet39 as the backbone to extract features.

>> -->>The vovnet39 is described in paper:An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection.the download link is :https://arxiv.org/abs/1904.09730

>> Jpu is used to get more semantic information that combine with vovnet39.
>> -->>The Jpu is described in paper:FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation.the download link is :http://export.arxiv.org/abs/1903.11816
## Environment
>> If you want to implement this project.the environment need build as follow:

>>>> python==3.6 

>>>> torch==1.1.0

>>>> numpy

>>>> matplotlib

>>>> tensorboardX

## Script interpret

>> The script dataprocess.py is for data read,it's actually a iterable.

>> The script metrics.py is defined miou.

>> The script vov_jpu.py is Vovnet39 combine the Jpu.

>> The script train.py is for train the model.

## Train 
>> I trained 120 epochs.bitch size is 8.
>> when you establish the environment,then can implement this project in terminal by "python train.py"

>> **Train loss and Train miou：**

>>>> ![Train loss and Train miou](images/1.png)

>> **Valid loss and Valid miou：**

>>>> ![Valid loss and Valid miou](images/2.png)

## Visual
>> **Segmentation result:**
![segmentation result1](save_visual/1.jpg)
![segmentation result2](save_visual/2.jpg)

## Analysis
>> There are still many improvements can use in this project, I just attempt to use separable convolution, dilate convolution and residual structure to unet. The result is better than original unet in my datasets. And the final use of the model is linear output, adding the sigmiod activate function may be better. The epoch of model training is less, and the effect of continuing training may be better.

## Attention
>> The project was completed by me independently for academic exchange. For commercial use, please contact me by email an_chao1994@163.com.
