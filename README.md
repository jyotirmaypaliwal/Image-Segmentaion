# Image-Segmentaion
#### I did image segmentation on cityscapes data taken from the below given link using UNet architecture. The dataset contained 2975 images in the training set and 500 images in the val set. 


Dataset - https://www.kaggle.com/dansbecker/cityscapes-image-pairs

#### The training images and their masks were combined in one image of dim 256&512 so we first had to separate those images. Also the masks were not in one hot encoded form so I first applied Kmeans to form 8 different clusters and then encoded the segmentation masks according to them. 

## UNet 

![1 f7YOaE4TWubwaFF7Z1fzNw](https://user-images.githubusercontent.com/27720480/137239380-84dc6694-3b19-4709-a797-b5bf4b0311d9.png)
#### The above given figure is the basic structure of the UNet model and we applied the same with some modifications. 

## Training
#### At first I trained the model using my own written UNet given in the file, "UNet.py" but it was not giving me the results I wanted. So I borrowed the model from https://github.com/jvanvugt/pytorch-unet which was working exceptionally well. For the optimizer, I used Adam and cross entropy loss as my loss function. 

#### I trained the model for a total of 50 epochs and training took about 2 hours.

## Result
#### The end result was quite satisfying and is shown below. 

<img width="441" alt="Screenshot 2021-10-14 075310" src="https://user-images.githubusercontent.com/27720480/137239904-d6b666ee-3122-4dc9-83f5-d6878c7f4fa1.png">

## Note
#### Feel free to contact me if you have any question regarding my project. 

