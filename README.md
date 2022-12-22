# Emotion Recognition using Convolutional Neural Networks

This project involves building a convolutional neural network (CNN) model for emotion recognition using the Face Expression Recognition dataset available on Kaggle.

## Tools and Libraries
- Python libraries for data cleaning and preprocessing: Matplotlib, Numpy, Pandas, Seaborn
* Deep learning library: Keras

## Dataset
[The Face Expression Recognition dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) consists of 48x48 pixel grayscale images of human faces with discrete labels for facial expressions. The human expressions are categorized into seven labels/categories: happy, fear, sad, neutral, disgust, surprise, and angry. The dataset is divided into training and validation sets, with 28821 and 7066 images, respectively.

## Model Building 
The CNN architecture of the model is quite simple; it consists of four convolution layers and two fully connected layers. For each convolution layer, we defined batch normalization and activation layers. The model is optimized using the Adam optimizer with a learning rate of 0.0001.

## Training and Validation
To fit the model with the training and validation data, we imported the ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau functions from the keras.callbacks module, and set the epochs to 48. We also used the early stopping function to stop the training of the model if the accuracy was not increasing with further epochs.

## Evaluation
We used the trained model to classify the images in the validation set, and calculated the classification accuracy and loss. We also plotted the classification accuracy and loss for each epoch to visualize the training and validation process.
