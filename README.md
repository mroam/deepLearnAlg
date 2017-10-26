# deepLearnAlg
Deep learning algorithms for video games
========
Griffin and Mike are trying to study machine learning by installing libraries, reading tutorials, running code, and revising code. Here is what we’ve found so far including links to installers, tutorials, versions, references, etc...


Potential Projects
-----------------
* 



References 
--------
* [TensorFlow!] (www.tensorflow.org) Machine-learning deep-learning framework, software library.
* [Anaconda](www.anaconda.com) Python Data-Science platform
* [Python](https://docs.python.org/3/) Python documents including [tutorials](https://docs.python.org/3/tutorial/index.html) and [faqs](https://docs.python.org/3/faq/index.html).
* [Python](https://docs.python.org/3/library/index.html) Python Standard Library Reference.


Dependencies
--------
* [TensorFlow!] (www.tensorflow.org/install/install_mac) 
* [homebrew] (brew.sh) Software package manager/updater, for mac. Gives us access to our other dependencies including...
* [numpy] (www.numpy.org) includes (we want!) scipy.org and matplotlib.
* [Anaconda](www.anaconda.com) Python Data-Science platform: supposedly manages the installation of the things like numpy, scipy, and matplotlib.
* pip package manager, used by Anaconda to install tensorflow. [Wikipedia re pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) says most distributions of python include pip. If it is not installed..."


Tutorials
--------
* [Udemy online course](http://www.udemy.com/deeplearning) $10 for "Deep Learning A-Z"
* Medium.com has some deeplearning tutorials, which we haven't tried yet, including [absolute beginners guide to machine learning](https://hackernoon.com/introduction-to-numpy-1-an-absolute-beginners-guide-to-machine-learning-and-data-science-5d87f13f0d51) which starts by installing numpy and teaching about numpy arrays


How To Work with Python in Anaconda
--------
AnacondaNavigator (is a mac app), has a home screen. Choose "spyder"'s "Launch" button. Suggestion: use menu "Run:ConfigurationPerFile" and set \[x]


To Do
--------
* Install tensorFlow after reviewing ownership of /usr/local/Cellar   suggestion sudo chown -R $(whoami) /usr/local/Cellar


Questions
--------
* Anaconda 5.0 can install tensorflow for us, but how sketchy is the "community" support of this install vs the "official" install of tensorflow? tensorflow.org has [page about using anaconda to install tensorflow](https://www.tensorflow.org/install/install_mac#installing_with_anaconda)
* How do we update Anaconda?


Questions that we've answered
--------
* How to install pip: Anaconda 5.0 installed python 3.6 including pip.
* How to install numpy: Anaconda 5.0 installed numpy.


Done
----------
* (√) Installed homebrew (required a sudo for installation, says that package downloads will NOT need sudo).
* (√) Installed Anaconda (did not need an admin password, offered to install python 3.6 and/or 2.7 (we installed 3.6)).
