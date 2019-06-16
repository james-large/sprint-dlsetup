# sprint-dlsetup
This is essentially a compilation of various semi-random links/instructions for setting up everything for Keras and sktime for the Alan Turing sprint event deep learning sub-project. 

The deep learning sub-project will largely involve interfacing with the networks defined in by [this](https://github.com/hfawaz/dl-4-tsc) repo, which are not in any kind of scikit learn format/structure. This will involve refactoring the code to the required format for compatibility with the rest of the sktime classifiers (to be decided/discovered during the event...), adding docs, and adding/performing tests. If this process turns out to be quite easy (dependant on language/code factors discussed later, time spent with head down coding during this week, and the expertise of those undertaking this project), we may implement networks from other sources of different kinds from scratch, however this is definitely a could rather than should or must.  

We will be implementing these with Keras and with a Tensorflow backend. In the future we can of course abstract away from Tensorflow and use whatever backend, but for the purposes of a sprint event, let's stick to using that. 

## Prerequisites 

Apologies, but a lot of these points will be most relevant for windows users, since that has always been my OS by default. For steps 4, 5, 6 (gpu, tensorflow, Keras setup), I strongly recommend following [this](https://github.com/antoniosehk/keras-tensorflow-windows-installation) guide for windows users. As far as I know the entire process should be easier for Linux users regardless. Mac users you're on you're own I'm afraid

1. [Python 3](https://www.python.org/downloads/) and the built-in package manager pip (should be installed with pythonm unless you chose a very old version), optionally [Anaconda](https://docs.anaconda.com/anaconda/install/windows/) for more automated virtual environment management, or can also be handled via pip alone just fine. 
2. Setup a virtual environment, up to you whether you want to use [virtualenv](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/) (via pip) or conda (via [command line](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) or the navigator GUI) etc. Numpy, Cython etc will be needed at minimum for sktime. see installation instructions link in next point. 
3. Sktime cloned and [installed](https://github.com/alan-turing-institute/sktime#installation). I have a fork [here](https://github.com/James-Large/sktime) with the implementation process started, but unsure how the Turing folks will want to structure the contributions over the course of this week. Side note - if you're using the PyCharm IDE, doing everything git-related within the IDE is probably easiest, can create the local project by cloning the repo automatically etc. 
4. Optional: CUDA and CUDNN if wanting to use a GPU (needed for anything other than basic testing on small data, really) 
5. Tensorflow via pip: Versioning in tensorflow is a pain. It seemingly isn't designed with backwards compatibility in mind at all, and different packages/libraries will work differently or not at all with different versions of Tensorflow. I'm not going to pretend I know about how everything interacts, rather I'll give the versions of everything that I have that can run a network to completion on my GPU below
6. Keras via pip: Should be easier. Again, don't know about new versions, but for Keras everything in 2.x is essentially back-compatible, and there's no reason I know of to use 1.x since we're dealing with new code. 

## My versioning

As said before, I'd argue to just follow the instructions [here](https://github.com/antoniosehk/keras-tensorflow-windows-installation) for what look like the most up-to-date versions of everything as of 2 months ago (I installed all my stuff a year+ ago), but that which crucially should work together. If all else fails, these are my versions of the finicky things: 

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
