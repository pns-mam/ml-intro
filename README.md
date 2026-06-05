![PNS](logo-pns.png)
# Introductory Machine Learning in Python
Course taught within a 1-week project @Polytech Sophia - MAM 3.

---- June 8th - June 12th 2026

## Instructors:

* [Mahmoud Elsawy](https://www-sop.inria.fr/atlantis/perso/Mahmoud.Elsawy/elsawy.html), mahmoud.elsawy@inria.fr
* [Jean-Luc Bouchot](jlbouchot.github.io), jean-luc.bouchot@inria.fr

## Course description

### Goals 
Gain a hands-on experience with solving a machine learning problem in Python.

By the end of the project students will have
* some basic knowledge of data visualisation
* some understanding of the importance of data visualisation
* some understanding of data reduction
* a first experience using scikit learn for a classification problem
* an understanding of pipelines, data leakage and overall training procedures
* a basic understanding of some mathematical aspects associated with ML (some optimisaton, modeling, algorithms)
* (ideally but not necessarily) experience with virtual environment and siloting

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
* skimage
* PCA
* support vector machines and kernel methods
* pipelines

### Progress and evaluation

The class will meet 8 times (4h a piece), the last meeting being dedicated to the evaluations
* Monday, Tuesday, Wednesday, Mornings 8:00 -- 12:15 and Afternoons 13:30 -- 17:30
* Thursday morning
* Evaluation on Friday morning

The project will be evaluated in groups of 3.

The evaluation will be done based on
* A small presentation
* A test code run of your work
* A report detailing your findings and choices made (**limited to 5 pages**)
* An analysis of all your code
All the deliverables should be sent to us by **Thursday 11/06/2026, 5pm**. 
Make sure to use our inria email addresses. 

In particular, throughout the course, you will be given scripts to fill out to guide you on your learning process. 
They should be sent to us before the evaluation. 


## Tentative plan

### Day 1: Intro and data preparation

* Project presentation, weekly organization
* Setting up the environment
* Documentation
* Train/Validation split and basic classification
* Playing with the data used for the rest of the class (PASCAL dog recognition dataset)
* PCA 
* Database visualisation

### Day 2: Classification with SVC and hyper parameter tuning

* train/test/validation split via cross validation
* Stratification
* One vs one, one vs rest
* pipelines
* support vector machines

### Day 3: More programming and tuning

* Hyperparameter optimisation
* Algorithm comparison

### Day 4: Time to wrap up

Make sure your report is done by the end of the business day

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
