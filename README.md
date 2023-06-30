# AsTeViR

AsTeViR is model-agnostic and can be adapted to various baseline methods. We provide our code based on 'Dual Encoding'.

## Environments

* **Ubuntu 16.04**
* **CUDA 11.1**
* **Python 3.8**
* **Pytorch 1.9.0**
* **Numpy 1.19.2**
* **Scipy 1.5.4**
* **Tensorboard-logger 0.1.0**

## Dataset

To run our code, please first download the required dataset and a pre-trained word2vec [here](https://drive.google.com/drive/folders/1TEIjErztZNQAi6AyNu9cK5STwo74oI8I). Then extract the content in ***code/dataset/***.

## Training

Run the script for training.

```shell
cd LINAS
./train_distill.sh $GPU_DEVICE $support-set-size-video $support-set-size-caption
```
The best performance can be reproduced with support-set-size-video as 10 and support-set-size-caption as 8.

## Evaluation

The evaluation on the test set will be performed automatically after the training. If you would like to evaluate before the end of training, please add the `--test` to the training script.
