# Cat Dog Classifier
This package contains modules for building a convolutional neural network cat classifier that I wrote for an intro to neural networks and CNNs talk. The package has a couple of models you can pre-load and train and it should be straight-forward to generalize to new architectures from what's there. Time permitting, there might be additional notebooks that go over model building, optimization, and regularizaiton. :)

Please note that the repository is ~80MB because of the training data that's included.

## Getting started
1. Clone the repository: 

```
git clone https://github.com/mattmeehan/catdog.git
```

2. Install the requirements

```
pip install -r requirements.txt
```

3. Package comes with 4000 pre-processed 50z50 grayscale images in ```./train_data/```. If you'd like to have more images, change the size, or train a model that utilizes all three color channels, see "Getting the data" below.

4. You're good to go! ```cat_classifier.ipynb``` should get you started.

## Getting the data
1. Make a Kaggle account so that you're able to access the data. You need to agree to particular terms to gain access to the data. 

2. Download data from https://www.kaggle.com/c/dogs-vs-cats
   Be aware that the data set is ~900 MB!

3. Unzip the compressed data files, which will give you two directories:
    
    train/
        Contains 25k images named cat.\*.jpg and dog.\*.jpg for training
    
    test1/
        Unlabled images that are meant for the Kaggle competition - we won't use these.

