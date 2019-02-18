# face-attribute-prediction
Face Attribute Prediction on [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) benchmark with PyTorch Implemantation, heavily borrowed from [my MobileNetV2 implementation](https://github.com/d-li14/mobilenetv2.pytorch).

## Dependencies

* Anaconda3 (Python 3.6+, with Numpy etc.)
* PyTorch 0.4+
* tensorboard, tensorboardX

## Dataset

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset is a large-scale face dataset with attribute-based annotations. Cropped and aligned face regions are utilized as the training source. For the pre-processed data and specific split, please feel free to contact me: <d-li14@mails.tsighua.edu.cn>

## Features

* Both ResNet and MobileNet as the backbone for scalability
* Each of the 40 annotated attributes predicted with multi-head networks
* Achieve ~92% average accuracy, comparative to state-of-the-art
* Fast convergence (5~10 epochs) through finetuning the ImageNet pre-trained models
