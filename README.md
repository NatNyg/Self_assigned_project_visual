# Self_assigned_project_visual
## This is the repository for my self assigned project for my visual analytics portfolio

### Description of project 
This script performs Transfer Learning on a dataset consisting of images labelled "good guy" or "bad guy". This means that I am using a pretrained convolutional neural network (VGG16) for feature extraction on my image data. Instead of normalizing by flattening and turning the images into black and white, this process turns the images into dense vector representations, thereby retaining more of the original image.The goal of the project is to use this transfer learning for a classification problem on the good guy/bad guy data, enabling the model to predict whether a person on a given image is in fact a good or a bad guy. 
This, of cause, does hold some biases and ethical considerations, of which the creator of the dataset has taken into consideration. These can be found on the following page https://www.kaggle.com/datasets/gpiosenka/good-guysbad-guys-image-data-set and reflects my use of the dataset as well as the creator's. 

### Method
This data had a different structure when downloaded from the webpage, than we have worked with on this course earlier, which means I had to do a bit of wrangling in order to work with it following the same methods that we have learned throughout the semester. 
Firstly, the data was split into test and train datasets beforehand, each folder containing two further subfolders; "savory" and "unsavory" (good-guys and bad-guys). In order to work with the data best possibly I started out by gathering all the images in the "savory" and "unsavory" folders, independent of the initial train/test split in two new folders; "good-guys" and "bad-guys", which I placed in a main folder; "in". After having readied my data I used the tensforflow keras function "image_dataset_from_directory" in order to access the subfolders in my main folder, and reading the data in the format of BatchDatasets into my workspace. This method further allowed me to split the data into train and test datasets. I tried normalizing my data, but this didn't give me a better result for the accuracy - in fact it gave me a slightly worse result, so I ended up going with the data without being normalized. 

After this I was able to fetch the labels from the images and binarizing them, using the sci-kit learn label binarizer function. After this the data is ready for the pretrained VGG16 model to perform the transfer learning. I then use keras from tensorflow to define the model and that I don't want the to train the layers, as this would defeat the purpose of the transfer learning. After having defined and saved my new model, I use Sci-kit learn to make a classification report, which is then saved to my "out" folder. I also save an image of the plotted accuracy and loss of the model on the training vs testing data.

### Usage
In order to reproduce my results, a few steps has to be followed:

1) Install the relevant packages - the list of the prerequisites for the script can be found in the requirements.txt
2) Make sure to place the script in the "src" folder. The data used in my code is fetched from https://www.kaggle.com/datasets/gpiosenka/good-guysbad-guys-image-data-set , wrangled and then put in the 'in' folder - so this folder can just be duplicated for reproducing. 
3) Run the script from the terminal.

This should give you the same results as I have gotten in the "out" folder.

### Results 
The results of this transfer learning solution to my classification task is actually really good - maybe even suspiciously so. This is something the author of the dataset also discusses in his entry on Kaggle, where he has tried to figure out just how the results are achieved by running the classification on individual images in order to trouble shoot / break down the process; "What bothers me is that the misclassification rate for individual images is significantly higher than one would expect based on the accuracy of he test set results. So something is going on but I don't know what. Thought it might be the emotion shown in the image but tests showed no clear correlation." (Gerry, https://www.kaggle.com/datasets/gpiosenka/good-guysbad-guys-image-data-set) 
I myself get an accuracy score of around 95% which is quite good, but the result also puzzled me, and even more so after reading the creator of the dataset wondering about this. However, the result in itself is really good, and the history plotted shows also that the model is not 100% steady all the way through, but is also not completely over- nor underfitting on the data. 

### Credits
For this project I have found inspiration in several pages, including:
https://www.tensorflow.org/tutorials/load_data/images 
https://stackoverflow.com/questions/64687375/get-labels-from-dataset-when-using-tensorflow-image-dataset-from-directory 


