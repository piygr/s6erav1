# Session 6 Assignment

Modle to detect handwritten digits, trained on MNIST dataset of 60,000 images.

**Goal is to create a model with**
- 99.4% validation accuracy
- Less than 20k Parameters
- Less than 20 Epochs
- Have used BN, Dropout,
- (Optional): a Fully connected layer, have used GAP.

## model.py
The file contains model class *OptimizedNet* as subclass of _torch.nn.Module_. The _OptimizedNet_ model has 3 convolution blocks, followed by GAP layer.

Below is the model summary -
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
              ReLU-3            [-1, 8, 28, 28]               0
         Dropout2d-4            [-1, 8, 28, 28]               0
            Conv2d-5            [-1, 8, 28, 28]             584
       BatchNorm2d-6            [-1, 8, 28, 28]              16
              ReLU-7            [-1, 8, 28, 28]               0
         Dropout2d-8            [-1, 8, 28, 28]               0
         MaxPool2d-9            [-1, 8, 14, 14]               0
           Conv2d-10           [-1, 16, 14, 14]           1,168
      BatchNorm2d-11           [-1, 16, 14, 14]              32
             ReLU-12           [-1, 16, 14, 14]               0
        Dropout2d-13           [-1, 16, 14, 14]               0
           Conv2d-14           [-1, 16, 14, 14]           2,320
      BatchNorm2d-15           [-1, 16, 14, 14]              32
             ReLU-16           [-1, 16, 14, 14]               0
        Dropout2d-17           [-1, 16, 14, 14]               0
           Conv2d-18           [-1, 32, 12, 12]           4,640
      BatchNorm2d-19           [-1, 32, 12, 12]              64
             ReLU-20           [-1, 32, 12, 12]               0
        Dropout2d-21           [-1, 32, 12, 12]               0
        MaxPool2d-22             [-1, 32, 6, 6]               0
           Conv2d-23             [-1, 32, 4, 4]           9,248
      BatchNorm2d-24             [-1, 32, 4, 4]              64
             ReLU-25             [-1, 32, 4, 4]               0
        Dropout2d-26             [-1, 32, 4, 4]               0
           Conv2d-27             [-1, 10, 4, 4]             330
================================================================
Total params: 18,594
Trainable params: 18,594
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.75
Params size (MB): 0.07
Estimated Total Size (MB): 0.83
----------------------------------------------------------------
```

## utils.py
The file contains utility & helper functions needed for training & for evaluating our model.

## S6.ipynb
The file is an IPython notebook. The notebook imports helper functions & _OptimizedNet_ model class from utils.py and model.py respectively.
In the notebook, we are creating train & test datasets with various transformations on the base MNIST dataset.

We can monitor our model performance while it's getting trained. The output looks like this - 
```
Epoch 1
Train: Loss=0.0829 Batch_id=937 Accuracy=87.58: 100%|██████████| 938/938 [00:41<00:00, 22.56it/s]
Test set: Average loss: 0.0524, Accuracy: 9856/10000 (98.56%)

Adjusting learning rate of group 0 to 1.0000e-02.

Epoch 2
Train: Loss=0.0588 Batch_id=937 Accuracy=96.93: 100%|██████████| 938/938 [00:41<00:00, 22.35it/s]
Test set: Average loss: 0.0335, Accuracy: 9904/10000 (99.04%)

Adjusting learning rate of group 0 to 1.0000e-02.

Epoch 3
Train: Loss=0.1722 Batch_id=937 Accuracy=97.58: 100%|██████████| 938/938 [00:43<00:00, 21.49it/s]
Test set: Average loss: 0.0292, Accuracy: 9916/10000 (99.16%)

Adjusting learning rate of group 0 to 1.0000e-02.

Epoch 4
Train: Loss=0.0434 Batch_id=937 Accuracy=97.79: 100%|██████████| 938/938 [00:42<00:00, 22.10it/s]
Test set: Average loss: 0.0270, Accuracy: 9910/10000 (99.10%)

Adjusting learning rate of group 0 to 1.0000e-02.

Epoch 5
Train: Loss=0.0943 Batch_id=937 Accuracy=97.97: 100%|██████████| 938/938 [00:42<00:00, 22.27it/s]
Test set: Average loss: 0.0218, Accuracy: 9927/10000 (99.27%)

Adjusting learning rate of group 0 to 1.0000e-02.

Epoch 6
Train: Loss=0.0167 Batch_id=937 Accuracy=98.23: 100%|██████████| 938/938 [00:43<00:00, 21.62it/s]
Test set: Average loss: 0.0207, Accuracy: 9941/10000 (99.41%)

Adjusting learning rate of group 0 to 1.0000e-02.

Epoch 7
Train: Loss=0.1589 Batch_id=937 Accuracy=98.27: 100%|██████████| 938/938 [00:42<00:00, 22.02it/s]
Test set: Average loss: 0.0186, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 1.0000e-03.

Epoch 8
Train: Loss=0.0111 Batch_id=937 Accuracy=98.62: 100%|██████████| 938/938 [00:42<00:00, 22.11it/s]
Test set: Average loss: 0.0164, Accuracy: 9945/10000 (99.45%)

Adjusting learning rate of group 0 to 1.0000e-03.

Epoch 9
Train: Loss=0.0019 Batch_id=937 Accuracy=98.64: 100%|██████████| 938/938 [00:42<00:00, 22.08it/s]
Test set: Average loss: 0.0162, Accuracy: 9951/10000 (99.51%)

Adjusting learning rate of group 0 to 1.0000e-03.

Epoch 10
Train: Loss=0.0296 Batch_id=937 Accuracy=98.77: 100%|██████████| 938/938 [00:42<00:00, 22.03it/s]
Test set: Average loss: 0.0159, Accuracy: 9948/10000 (99.48%)

Adjusting learning rate of group 0 to 1.0000e-03.

Epoch 11
Train: Loss=0.0816 Batch_id=937 Accuracy=98.76: 100%|██████████| 938/938 [00:45<00:00, 20.47it/s]
Test set: Average loss: 0.0155, Accuracy: 9952/10000 (99.52%)

Adjusting learning rate of group 0 to 1.0000e-03.

Epoch 12
Train: Loss=0.0255 Batch_id=937 Accuracy=98.65: 100%|██████████| 938/938 [00:46<00:00, 20.19it/s]
Test set: Average loss: 0.0152, Accuracy: 9949/10000 (99.49%)

Adjusting learning rate of group 0 to 1.0000e-03.

Epoch 13
Train: Loss=0.0262 Batch_id=763 Accuracy=98.72:  81%|████████  | 762/938 [00:43<00:08, 20.67it/s]
```  

## How to setup
### Prerequisits
```
1. python 3.8 or higher
2. pip 22 or higher
```

It's recommended to use virtualenv so that there's no conflict of package versions if there are multiple projects configured on a single system. 
Read more about [virtualenv](https://virtualenv.pypa.io/en/latest/). 

Once virtualenv is activated (or otherwise not opted), install required packages using following command. 

```
pip install requirements.txt
```

## Running IPython Notebook (S6.ipynb) using jupyter
To run the notebook locally -
```
$> cd <to the project folder>
$> jupyter notebook
```
The jupyter server starts with the following output -
```
To access the notebook, open this file in a browser:
        file:///<path to home folder>/Library/Jupyter/runtime/nbserver-71178-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
     or http://127.0.0.1:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
```

Open the above link in your favourite browser, a page similar to below shall be loaded.

![Jupyter server index page](https://github.com/piygr/s5erav1/assets/135162847/40087757-4c99-4b98-8abd-5c4ce95eda38)

- Click on the notebook (.ipynb) link.

A page similar to below shall be loaded. Make sure, it shows *trusted* in top bar. 
If it's not _trusted_, click on *Trust* button and add to the trusted files.

![Jupyter notebook page](https://github.com/piygr/s5erav1/assets/135162847/7858da8f-e07e-47cd-9aa9-19c8c569def1)
Now, the notebook can be operated from the action panel.

Happy Modeling :-) 
 
