# Cat Dog Classifier
This is a brief introduction to deep learning using the Kaggle cat vs. dog data set. The package has a couple of models you can pre-load and train, but it should be easy to generalize to new architectures from what's there. Time permitting, there might be additional notebooks that go over model building, optimization, and regularizaiton. :)

## Getting started
Clone the repository: 

```
git clone https://github.com/mattmeehan/catdog.git
```

## Getting the data
1. Make a Kaggle account so that you're able to access the data. You need to agree to particular terms to gain access to the data. 

2. Download data from https://www.kaggle.com/c/dogs-vs-cats
   Be aware that the data set is ~900 GB!

3. Unzip the compressed data files, which will give you two directories:
    train/
        Contains 25k images named cat.\*.jpg and dog.\*.jpg for training
    test1/
        Unlabled images that are meant for the Kaggle competition - we won't use these.

