# Machine Learning in Python
Course taught within a 1-week project @Polytech Sophia - MAM 3.

---- June 10th - June 13th 2025

Instructors:
* Mahmoud Elsawy, mahmoud.elsawy@inria.fr
* Jean-Luc Bouchot, jean-luc.bouchot@inria.fr

## Course description

### Goals 
Gain a hands-on experience with solving a machine learning problem in Python.

By the end of the project students will have
* experience with virtual environment and siloting
* some basic knowledge of data visualisation
* some understanding of data reduction
* a first experience using scikit learn for a classification problem
* a basic understanding of some mathematical aspects associated with ML (backpropagation, optimisaton, modeling)

What this course is not about:
* A theoretical computer science course. While the implementation part is important to a successful project, the quality of the code is not is the main part of the assessment. 
* a full theoretical description of the machine learning landscape

### Topics
Here are some keywords related to the course. Some of them will be given great details throughout the week, some others are here for the curious and interested reader
* Constrained lagrangian optimisation 
* Regression vs classification 
* 1v1, 1vAll
* virutal environments
* scikitlearn
* torch 
* tensorflow
* neural networks
* convolutional neural networks
* deep learning 


### Progress and evaluation

The class will meet 6 times (5 x 4h + 6h), the last meeting being dedicated to the evaluations
* Mornings of Tu,W,Th, 8am 12 noon
* Afternoons of Tu,W, 130pm-530pm
* Evaluation on Friday afternoon 1pm-7pm

The project will be evaluated in groups of 3, randomly chosen.

The evaluation will be done based on
* A small presentation
* A test code run of your work
* A report detailing your findings and choices made

In particular, throughout the course, you will be given 3 scripts to fill out to guide you on your learning process. 
They should be sent to us before the evaluation. 
But the evaluation will use a 4 small script containing your final algorithmic choices. 
The outcome of this final script should be a model on which a `predict` and/or a `score` function can be applied and which we will use on one of our databases.
This will be further clarified in class. 


## Tentative plan

### Day 1: Intro and data preparation

* Project presentation, weekly organization
* Setting up the environment
* Documentation
* Train/Validation split and basic classification
* Playing with the data used for the rest of the class (handwritten digits)
* PCA 
* Database visualisation

### Day 2: Classification with SVC and hyper parameter tuning

* train/test/validation split via cross validation
* Stratification
* One vs one, one vs rest
* pipelines

### Day 3: Neural networks and final tuning

* Activation function
* Optimisers (Adam, sgd,...)
* Fully connected NN
* Report and final algo

## Ressources

### Setting up your environment

First clone this repository

> git clone git@github.com:pns-mam/ml-intro.git 
> cd ml-intro

Then create a virtual environment to contain your work. 
We recommend reading [this documentation](https://packaging.python.org/en/latest/tutorials/installing-packages/) to learn more about all this. (Ou encore [cette page](https://docs.python.org/fr/dev/installing/index.html) pour les lecteurs francophones)
(current description tested on Linux. Adaptations needed for windows users)

You can find ressources [on venv](https://docs.python.org/3/library/venv.html)
> python -m venv MLPythonVenv

Once your (empty) environment has been created, activate it 
> source MLPythonVenv/bin/activate

You can always deactivate this virtual environment by simply typing 
> deactivate 

We can start adding useful packages to this environment using the `pip install SomePackage` command

Here is a list of packages which you may want to add to your environment
* jupyter (notebooks)
* seaborn (relatively nice plots)
* pandas (package for handling tabular data)
* scikit-learn (main package for machine learning)
* skimage (some useful image processing routines)


