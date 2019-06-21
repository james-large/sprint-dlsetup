# sprint-dlsetup
This is essentially a compilation of various semi-random links/instructions for setting up everything for Keras and sktime for the Alan Turing sprint event deep learning sub-project, and a setting out of the aims and intentions of it. 

## Project Intentions

The deep learning sub-project will initially and mainly involve interfacing with the networks defined in by [this](https://github.com/hfawaz/dl-4-tsc) repo, which are not in any kind of scikit learn format/structure. This will involve refactoring the code to the required format for compatibility with the rest of the sktime classifiers (to be decided/discovered during the event...), adding docs, and adding/performing tests. 

As a result, the first set of classifiers implemented will be particular, pre-defined architectures of various kinds. For example, one of the networks if a simple Fully Connected Network (FCN), i.e. consecutive dense layers where the outputs of each layer connect to all the inputs of the next. In the [source paper](https://link.springer.com/article/10.1007%2Fs10618-019-00619-1), a particular number of layers, nodes in each layer, etc. are defined which we shall interface with exactly. 

Further, the tuning of network hyper-parameters like batch size, dropout, is a possible via the KerasClassifier wrapper and GridsearchCV in scikitlearn (see below) which we could also emulate this week. 

This is for a couple of reasons. First, confirming correctness of the interfacing via recreation of some results is obviously easier if we maintain architectures. Second, the source paper represents a decent anchor point for a comprehensive comparative evaluation for deep learning within TSC in particular. While it has of course taken over computer vision and various image classification tasks, it has not yet overtaken TSC, but maybe this serves as the start of it. Having access to these predefined architectures to run in it without expertise knowledge is a very good thing for interested users. 

If this process turns out to be quite easy (dependant on language/code factors discussed later, time spent with head down coding during this week, and the expertise of those undertaking this project), we may implement networks from other sources of different kinds from scratch, however this is definitely a could rather than should or must. See the [project page](https://github.com/alan-turing-institute/sktime/projects/5) for a basic [MoSCoW](https://en.wikipedia.org/wiki/MoSCoW_method0) of this week.

A possible outcome, which may be born naturally from how we undertake the project, is to implement a template class for a Keras network, such that a future user/contributor to sktime need only define any particular data preprocessing and a build_network function - defining the core network architecture - to include their own network architectures in any grander pipelines they want to use in sktime. 

We will be implementing these with Keras and with a Tensorflow backend. In the future we can of course abstract away from Tensorflow and use whatever backend, but for the purposes of a sprint event, let's stick to using that. 

One factor which I'm not sure on the possible outcome of is making the installing and usage of Keras/Tensorflow as easy as possible (and/or optional, when not intending to use these networks) in terms of dependancy management within user's environments etc. We'll have to have a look at how cython dependancies have been handled and see what similar systems we can implement for this, however CUDNN is almost certainly something that each user would have to install themselves if gpu usage is required.


## Prerequisites 

Apologies, but a lot of these points will be most relevant for windows users, since that has always been my OS by default. For steps 4, 5, 6 (gpu, tensorflow, Keras setup), I strongly recommend following [this](https://github.com/antoniosehk/keras-tensorflow-windows-installation) guide for windows users. As far as I know the entire process should be easier for Linux users regardless. Mac users you're on you're own I'm afraid

1. [Python 3](https://www.python.org/downloads/) and the built-in package manager pip (should be installed with pythonm unless you chose a very old version), optionally [Anaconda](https://docs.anaconda.com/anaconda/install/windows/) for more automated virtual environment management, or can also be handled via pip alone just fine. See the sktime team's own [instructions](https://github.com/alan-turing-institute/sktime/wiki/2019-sktime-MLJ-tutorial-development-sprint#getting-started) for more info. 
2. Setup a virtual environment, up to you whether you want to use [virtualenv](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/) (via pip) or conda (via [command line](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) or the navigator GUI) etc. Numpy, Cython etc will be needed at minimum for sktime. see installation instructions link in next point. 
3. Sktime cloned and [installed](https://github.com/alan-turing-institute/sktime#installation). I have a fork [here](https://github.com/James-Large/sktime) with the implementation process started, but unsure how the Turing folks will want to structure the contributions over the course of this week. Side note - if you're using the PyCharm IDE, doing everything git-related within the IDE is probably easiest, can create the local project by cloning the repo automatically etc. 
4. Optional: CUDA and CUDNN if wanting to use a GPU (needed for anything other than basic testing on small data, really) 
5. Tensorflow via pip: Versioning in tensorflow is a pain. It seemingly isn't designed with backwards compatibility in mind at all, and different packages/libraries will work differently or not at all with different versions of Tensorflow. I'm not going to pretend I know about how everything interacts, rather I'll give the versions of everything that I have that can run a network to completion on my GPU below
6. Keras via pip: Should be easier. Again, don't know about new versions, but for Keras everything in 2.x is essentially back-compatible, and there's no reason I know of to use 1.x since we're dealing with new code. 

## My versioning

As said before, I'd argue to just follow the instructions [here](https://github.com/antoniosehk/keras-tensorflow-windows-installation) for what look like the most up-to-date versions of everything as of 2 months ago (I installed all my stuff a year+ ago, and am following a 'aint broke, dont fix it' strategy), but that which crucially should work together. If all else fails, these are my versions of the finicky things: 

* CUDNN                  9.2
* tensorflow-gpu         1.8.0     
* Keras                  2.2.0

## Contributing to sktime

This part is still somewhat up in the air, however a few things of note are: 

* The most likely way we'll cleanly interface with everything is via the [KerasClassifier wrapper](https://keras.io/scikit-learn-api/), which makes Keras models usable with scikit learn stuff like GridSearchCV. Unfortunately, the restriction with this is supposedly that only [Sequential](https://keras.io/getting-started/sequential-model-guide/) models, as opposed to 'Model's themselves made via the [functional API](https://keras.io/getting-started/functional-api-guide/), can be used to interface with. Broadly speaking, this means that only models with single inputs, single outputs, can be wrapped. This is fine for most of the networks in dl-4-tsc, however things like [ResNet](https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py) may be a pain in modelling the residual connections. We'll discover the limits of this system over this week. Perhaps there are hacks via stuff like duck-typing we can do, not sure. That's for people with better python knowledge than me to figure out. 

* In terms of code structure and so on, it appears that the sktime group has a BaseClassifier class that inherits from scikit's BaseEstimator, and acts like an alternative for the BaseEstimator and ClassiifierMixin inheritance structure that would be used [when contributing to scikit directly](https://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator). Broadly though, a lot of the same principles apply: 
  * All parameters of __init__ should has default arguments
  * fit, predict, and predict_proba should all be overridden
  * etcetc 

Essentially, I think we'll find out the exact requirements for compatibility once we sit down and have a chat during the week. In general, if it works with scikit learn, it should at the very least be very quick to refactor into working with sktime. Further, refactoring and so on can happen after this week - this is a sprint after all.

## Working(?) example, conversion stages

I've setup a [fork](https://github.com/James-Large/sktime) of sktime, where I've [branched off of dev](https://github.com/James-Large/sktime/tree/dl4tsc/sktime/classifiers), and started adding the converted network fcn, in couple different formats. As with the user testing session on Monday, you definitely want to be branching off of dev instead of master in general when workign with sktime. 

The original author, Hassan Fawaz, and I have started with a basic conversion of the FCN network, [original](https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py), direct conversion [number one](https://github.com/James-Large/sktime/blob/dl4tsc/sktime/contrib/deeplearning_based/fcn_fawaz.py) with KerasClassifier as the super class, and attempted conversion [number two](https://github.com/James-Large/sktime/blob/dl4tsc/sktime/contrib/deeplearning_based/fcn_noKC.py) with sktime's BaseClassifier as the super class.

Broadly, Keras models have three main parts to their definition: 

* The input shape (series length, number of dimensions, batch size). The shape of all standard later layers are inferred from the previous in Keras
* The architecture (number and type of the layers etc.)
* The parameters of the training/convergence process; the loss function, any validation metrics, and the optimisation algorithm (e.g. gradient descent, Adam optimiser etc.)

Callbacks that effect the fit process can also be added and various other addons, but these three are the main aspects. 

## Outcomes

### Tuesday, 18/06/2019 

* Contributors: James Large, Aaron Bostrom
* Decided on a basic format for the conversion - not using KerasClassifier directly but using some functionality from it
* Made a base class for the conversions, BaseDeepLearner, which inherits from BaseClassifier. Has some functionality for one-hot-encoding labels 
* Set up a basic testbed for compatibility, one which simply loads data, constructs classifier, fits and scores, and one which does the same but within a basic pipeline object. A third, using the high-level time series tasks and strategy interfaces were not working for as yet unknown reasons, class labels within the networks were numerical of course, but still reamining as strings in the high-level interface for comparing true and predicted labels
* Converted a number of the networks in dl-4-tsc, all of the simpler ones (simpler as in without extra preprocessing or augmentation steps outside of the actual keras model definition) and converted two of the more difficult examples. Fully testing these will require an edit to the base learner's predict methods
* Started experiments to run overnight with those completed networks to confirm parity between published and reproduced results/networks

### Wednesday, 18/06/2019 

* Experiments from day before did not work, wasn't compatible with something later in the experimental pipeline
* Fixed this error, and finished implementation of networks. MCNN, however, had and error where (we believe) it was running out of memory without reporting it. To be fixed tomorrow
* All networks aside from MCNN passing basic run tests, running experiments again over night

### Thursday, 18/06/2019 

* Refactored static dl-4-tsc networks into own directory
* Finshed off conversion/fixes, however a memory leak is potentially still aroudn. Currently clearing backend/garbage collecting every model/experiment run
* Fixed the label encoding issue, the networks now work with high-level sktime functionality, TSCTask, TSCStrategy etc
* Got started on implementing tunable networks

### Friday, 18/06/2019 

* stuff
* more stuff

## Beyond
