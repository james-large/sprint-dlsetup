# sprint-dlsetup
Various semi-random links/instructions for setting up everything for keras and sktime for the Turing sprint event

The deep learning sub project will largely involve interfacing with the networks defined in by [this](https://github.com/hfawaz/dl-4-tsc) repo, which are not in any kind of scikit learn format/structure. This will involve refactoring the code to the required format for compatibility with the rest of the sktime classifiers (to be decided/discovered during the event...), adding docs, and adding/performing tests. If this process turns out to be quite easy (dependant on language/code factors discussed later, time spent with head down coding during this week, and the expertise of those undertaking this project), we may implement networks from other sources of different kinds from scratch, however this is definitely a could rather than should or must.  

We will be implementing these with Keras and with a Tensorflow backend. In the future we can of course astract away from Tensorflow and use whatever backend, but for the purposes of a sprint event, let's stick to using that. 

## Prerequisites 

For steps 4, 5, 6 (gpu, tensorflow, Keras setup), I strongly recommend following [this](https://github.com/antoniosehk/keras-tensorflow-windows-installation) guide for windows users. As far as I know the entire process should be easier for Linux users regardless. max users you're on you're own I'm afraid

1. Python 3.x, pip, of course
2. Setup a pip environment, up to you whether you want to use virtualenv or conda etc. Numpy, Cython etc will be needed at minimum for sktime. see installation instructions link in next point. 
3. Sktime cloned and [installed](https://github.com/alan-turing-institute/sktime#installation). I have a fork [here](https://github.com/James-Large/sktime) with the implementation process started, but unsire how the Turing folks will want to structure the contributions over the course of this week. Side note - if you're using the PyCharm IDE, doing everythin git-related within the IDE is probably easiest, can create the local project by cloning the repo automatically etc. 
4. Optional: CUDA and CUDNN if wanting to use a GPU (needed for anything other than basic testing on small data, really) 
5. Tensorflow via pip: Versioning in tensorflow is a pain. It semmingly isn't designed with backwards compatibility in mind at all, and different packages/libraries will work differently or not at all with different versions of Tensorflow. I'm not going to pretend I know about how everything interacts, rather I'll give the versions of everything that I have that can run a network to completion on my GPU below
6. Keras via pip: Should be easier. Again, don't know about new versions, but for Keras everything in 2.x is essentially back-compatible, and there's no reason I know of to use 1.x since we're dealing with new code. 

## My versioning

As said before, I'd argue to just follow the instructions [here](https://github.com/antoniosehk/keras-tensorflow-windows-installation) for what look like the most up-to-date versions of everything as of 2 months ago, but that which cruicially should work together. If all else fails, these are my versions of the finicky things: 

* tensorflow-gpu         1.8.0     
* Keras                  2.2.0

