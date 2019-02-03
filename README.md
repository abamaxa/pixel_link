# Pixel Link using a Mobilenet feature detector

This repository contains a modified version of the [PixelLink](https://github.com/ZJULearning/pixel_link) text detector. This fork uses a Mobilenet v2 feature detector, rather than the VGG detector used in the original version. Please see the original repository for more details.

The advantage of using Mobilenet instead of ResNet for feature detection is a substantial reduction in model size for a relatively small drop in accuracy. Furthermore, by using deconvolution transforms instead of bilinear interpolation for upsampling in the fusion layers, this model slightly outperforms the original design.

# Usage

##Training

As per the original, this model is intended to be trained on the Synthtext and icdar2015 image sets. A modified version of the training script can be found in the scripts subdirectory. 

The training script expects the training data to be available in a subdirectory called "training_data". 

A machine with a GPU, supported by Tensorflow, with a minimum of 8GB RAM is recommended for training. At the time of writing, it is not practicable to train this model on a CPU alone.

The script takes the following parameters:

|Parameter|Description|
|---|---|
|GPUS|zero indexed list of GPUs to use.|
|Batch size|the training batch size, 8 is recommended for GPUs with 8GB RAM|
|Output Directory|the directory to save the trained model and checkpoints to.|

### Examples

To train a model using GPUs 0 and 1, with a batch size of 8, saving the model to a subdirectoru called model, execute the following:

```shell
pixel_link$ scripts/train.sh 0,1 8 model
```

## Freezing the Model

A training checkpoint can be frozen for optimal use for inference by using the freeze.py script, found in the top level directory. 

Frozen models are saved in a subdirectory named "frozen".

So, to freeze checkpoint number 372284 in the model directory, saving the frozen model as frozen/mymodel.pb, execute the following:

```shell
pixel_link$ python freeze.py \
	--eval_image_height=1152 \
	--eval_image_width=1536 \
	--model_name=tiny \
	--pixel_conf_threshold=0.5 \
	--link_conf_threshold=0.5 \
	--checkpoint_path=model/model.ckpt-372284 \
	--output=mymodel
``` 

## Converting the frozen model to CoreML format

CoreML is Apples' machine learning library. It provides access to hardware accelerated inference on recent iPhone models.  

The ios.py script can be used to convert a trained frozen model to CoreML format. 

The script takes the name of the frozen model, saved in the frozen subdirectory, and saves the converted model to the same directory, with a .mlmodel extension. 

So to convert a frozen model, frozen/mymodel.pb, execute the following:

```
pixel_link$ python ios.py mymodel.pb
```

The converted model will be saved as frozen/mymodel.mlmodel.

# Inference Library

Neither Tensorflow nor CoreML provide all the operations required to carry out inferences using the model. In the original source, inference calculations are performed using python routines. 

In order to allow inferences to be performed on iOS or Android, the inference code has been rewritten in C++. This new inference code is contained in the post_process subdirectory.

# TODO

Basic sanity testing and sample code for an iOS app using model.

# Licence

This fork is released under the same MIT licence as the original.