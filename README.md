# sprint-dlsetup
Various semi-random links/instructions for setting up everything for keras and sktime for the Turing sprint event

The deep learning sub project will largely involve interfacing with the networks defined in by [this](https://github.com/hfawaz/dl-4-tsc) repo, which are not in any kind of scikit learn format/structure. This will involve refactoring the code to the required format for compatibility with the rest of the sktime classifiers (to be decided/discovered during the event...), adding docs, and adding/performing tests. If this process turns out to be quite easy (dependant on language/code factors discussed later, time spent with head down coding during this week, and the expertise of those undertaking this project), we may implement networks from other sources of different kinds from scratch, however this is definitely a could rather than should or must.  

We will be implementing these with Keras and with a Tensorflow backend. In the future we can of course astract away from Tensorflow and use whatever backend, but for the purposes of a sprint event, let's stick to using that. 

Prerequisites therefore are: 

* Python 3.x, pip, of course
* Setup a pip environment, up to you whether you want to use virtualenv or conda etc. Numpy, Cython etc will be needed at minimum for sktime. see installation instructions link in next point. 
* Sktime cloned and [installed](https://github.com/alan-turing-institute/sktime#installation). I have a fork [here](https://github.com/James-Large/sktime) with the implementation process started, but unsire how the Turing folks will want to structure the contributions over the course of this week. Side note - if you're using the PyCharm IDE, doing everythin git-related within the IDE is probably easiest, can create the local project by cloning the repo automatically etc. 
* Optional: CUDA and CUDNN if wanting to use a GPU (needed for anything other than basic testing on small data, really) 
* Tensorflow via pip: Versioning in tensorflow is a pain. I'm not going to pretend I know about how everything interacts, rather I'll give the versions of everything that I have that can run a network to completion on my GPU below
* Keras via pip: Should be easier. Again, don't know about new versions, but for Keras everything in 2.x is essentially back-compatible, and there's no reason I know of to use 1.x since we're dealing with new code. 
