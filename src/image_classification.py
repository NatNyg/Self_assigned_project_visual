"""
First, let's import the libraries used for this script!
"""
# base tools
import os

# data analysis
import numpy as np

# tensorflow
import tensorflow as tf
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense)

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

# generic model object
from tensorflow.keras.models import Model

#sklearn
from sklearn.metrics import classification_report


import matplotlib.pyplot as plt

def plot_history(H, epochs):
    """
This function plots the history of the model, visualizing the loss and accuracy for the test and train data, and saves the plot to the "out" path. 
    """
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    plt.savefig(os.path.join("out","history.png"))


def load_data():
    """
This function loads my data that is placed in the "in" folder. I am using the tensorflow keras image_dataset_from_directory in order to load my data as a train and validation datasets, since my folder consists of two subfolders; "good-guy" and "bad_guy", which is pre-labelled images divided into the two classes. Lastly I fetch the labels for each image in the training and validation dataset.  
    """
    #defining a path to the directory of our data-folder that contains subfolders 
    data_dir = os.path.join("in")
    #function from tensorflow that takes the images from the subfolders in the given directory and transforms 
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123)
    class_names = train_ds.class_names
    train_label = np.concatenate([y for x, y in train_ds], axis=0)
    val_label = np.concatenate([y for x, y in val_ds], axis=0)
    return train_ds, val_ds, val_label, class_names
    
def define_model():
    """
This function defines the model I will be using for transfer learning (VGG16), turns of the trainable layers in order to use transfer learning and adds the new classifier layers using the sigmoid activiationlayer, since I'm working with a binary classification task. Lastly I define that I want to use a binary loss function, again as to match the binary classification on the data, and that I want to optimize the model for accuracy. I use the schedule part to define that I want to take "big steps" at first, and then slow down as the model is training. I'm using the stochastic gradient descent algorithm for the model.        
    """
    tf.keras.backend.clear_session()
    
    model = VGG16(include_top = False,
                  pooling = "avg",
                  input_shape = (256,256,3))
    for layer in model.layers:
        layer.trainable = False
    #add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu')(flat1)
    output = Dense(1, activation='sigmoid')(class1)

    #define new model 
    model = Model(inputs=model.inputs,
                  outputs=output)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9) #1 would be no change, so we want a small decrease here

    sgd =SGD(learning_rate=lr_schedule)
    model.compile(optimizer=sgd,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model 

def evaluate_model(model, train_ds, val_ds, val_label, class_names):
    """
This function firstly fits the just defined model and trains it on the trainingdata and saves the history of the training as an image to the "out" folder. It then uses the model to make predictions on the validation data and saves a classification report to the "out" folder. 
    """
    H = model.fit(train_ds,
                  validation_data = (val_ds),
                  batch_size = 128,
                  epochs = 10,
                  verbose = 1) 
    plot_history(H,10)
    predictions = model.predict(val_ds, batch_size = 128)

    preds = ["good_guy" if i>0.5
             else "bad_guy" for i in predictions]
    y_t = ["good_guy" if i>0.5 
          else "bad_guy" for i in val_label]
    report =classification_report(y_t,
                                  preds,
                                  target_names=class_names)
    with open('out/classification_report.txt', 'w') as file:
        file.write(report)
    return report
    
    
def main():
    """
The main function defines which functions to run, when the script is run from the terminal. 
    """
    train_ds, val_ds, val_label, class_names = load_data()
    model = define_model()
    report = evaluate_model(model, train_ds, val_ds, val_label, class_names)
    
    
if __name__== "__main__":
    main()
