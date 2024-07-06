## BraTS-Unet2D
**Project description**: Built for the brain tumour segmentation task, the data can be from any year of the BraTS challenge (the project used data from 2019) and model == Unet-2D
### You can find BraTS data here:
**_Website_**:
1. BraTS 2021-Kaggle (https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)
2. BraTS 2019 (https://www.med.upenn.edu/cbica/brats2019/data.html)

### 1. data processing

> Locate the data folder and run the [data_split.py](data/data_split.py)

Divide the dataset into trainset and testset and make a copy of it and save the divisions with the filename of the json file.

> Run [data_process.py](data/data_process.py)

Images saved in npz format, Ground Truth saved as png

### 2. Get JSON file
> Locate the home folder and run [get_json.py](get_json.py)

Get the json data files for train and test, data reading will be done based on these files.

### 3. train && test
Before you start running train, perhaps you need to confirm the args argument in train.py.
AND then in terminal: 
> python train.py

OR in pycharm
> RUN train

**To avoid trouble, the test function has also been written into the train.py file, which you can easily find and make changes to.**

*Simple? Aha!*

### Reference
**_Papers_**:
1. U-Net: Convolutional Networks for Biomedical Image Segmentation (https://arxiv.org/abs/1505.04597)
2. The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification (https://arxiv.org/abs/2107.02314) 
