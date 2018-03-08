# deepLearnAlg
## Deep learning algorithms for video games
Griffin and Mike are trying to study machine learning by installing libraries, reading tutorials, running code, and revising code. Here is what we’ve found so far including our notes (below and [numpyStackTutorial](numpyStackTutorial.md) and [tensorflow](tensorFlowOurNotes.md)) and links to installers, tutorials, versions, references, etc...


## Potential Projects
* Train an AI to play games.
* Train simulated cars to deal with pedestrians and traffic. [NYTimes article](https://www.nytimes.com/2017/10/29/business/virtual-reality-driverless-cars.html)
* Train AI to recognize twitterbots?



## References 
* [TensorFlow!](http://www.tensorflow.org) Google's big deal machine-learning deep-learning framework and software library.
* [Anaconda](http://www.anaconda.com) Python Data-Science platform, installs many things for us including python and tensorflow.
* [Kaggle](https://www.kaggle.com) Lots of info about machine learning including datasets. From Google.
* [openAI](https://www.openai.com) From Musk & Co., lots of machine learing including gym and universe. See our notes about installing them (below).
* [Python](https://docs.python.org/3/) Python documents including [tutorials](https://docs.python.org/3/tutorial/index.html) and [faqs](https://docs.python.org/3/faq/index.html).
* [Python](https://docs.python.org/3/library/index.html) Python Standard Library Reference.
* [Testing](http://docs.python-guide.org/en/latest/writing/tests/) Testing in Python!
* [Markdown](https://help.github.com/articles/about-writing-and-formatting-on-github/) how to edit “markdown” text on github.


## Related Articles
* [AlphaZero learns chess in 4 hours, beats world's favorite chess AI](https://www.chess.com/news/view/google-s-alphazero-destroys-stockfish-in-100-game-match)
* [INVIDIA wants to train youth in AI](https://www.technologyreview.com/the-download/609284/nvidia-is-aiming-to-train-the-next-generation-of-ai-experts/)
* [Different stick figures learn to walk](http://www.goatstream.com/research/papers/SA2013/)


## Related Videos
* [game cars learn to drive around a track](https://youtu.be/BhsgLeY_Q-Y)
* [simulated runners with obstacles](https://youtu.be/g59nSURxYgk)
* [Different stick figures learn to walk](https://youtu.be/pgaEE27nsQw)
* [openAI gym](gym.openai.com/envs/#classic_control) and [openAI walkers](gym.openai.com/envs/#mujoco)


## Dependencies
* [Anaconda](www.anaconda.com) Python Data-Science platform: supposedly manages the installation of the things like tensorflow, python, numpy, scipy, and matplotlib.
* [TensorFlow!](www.tensorflow.org/install/install_mac) 
* ?? [homebrew](http://brew.sh) Software package manager/updater, for mac. Is one way to get our other dependencies including numpy and scipy, but maybe Anaconda is taking care of these dependencies for us? Homebrew seems necessary for installing all of gym.openai.com onto a mac.
* [numpy](www.numpy.org) includes (we want!) scipy.org and matplotlib.
* pip package manager, used by Anaconda to install tensorflow. 
[Wikipedia re pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) says most distributions of python include pip. If it is not installed...."


## Tutorials
* [Udemy online course](http://www.udemy.com/deeplearning) $10 for "Deep Learning A-Z" has pre-requisites including 
   * [numpy stack](https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python/) . See [our notes](https://github.com/mroam/deepLearnAlg/blob/master/numpyStackTutorial.md) 
   * https://www.udemy.com/data-science-linear-regression-in-python/
* Medium.com has some deep-learning tutorials, which we haven't tried yet, including 
  * [absolute beginners guide to machine learning](https://hackernoon.com/introduction-to-numpy-1-an-absolute-beginners-guide-to-machine-learning-and-data-science-5d87f13f0d51) which starts by installing numpy and teaching about numpy arrays.
* [Getting started with tensorflow](https://www.tensorflow.org/get_started/get_started) has link to tensorflow starter tutorial.



## How To Work with Python in Anaconda
Anaconda-Navigator (is a mac app), and has a home screen in which we click "spyder"'s **"Launch"** button. Suggestion: use menu "Run:ConfigurationPerFile" and set \[x]ClearAllVariablesBeforeExecution.


## To Do
* Finish [installing gym.openai.com](github.com/openai/gym#basics) (began with `brew update` and then `brew install` but then we're finding that they want /usr/local/bin to be writable!!?? Why aren't they using ~/usr/local/bin for Pete's sake--see [brew docs](https://docs.brew.sh/FAQ)
* Study tutorials: we're part way into [notes numpy stack tutorial](https://github.com/mroam/deepLearnAlg/blob/master/numpyStackTutorial.md) at https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python/
* Install [ipython](https://ipython.org/) and [jupyter](https://jupyter.org/) (iPython runs interactive python inside the [jupyter](https://jupyter.org/) shell and notebook system) if necessary for the numpy stack tutorial. <== wait, did Anaconda install these for us??
* Continue trying to install tensorflow on ppp1d trying to follow  https://www.tensorflow.org/install/install_mac#installing_with_anaconda. Got through step 4, "conda" runs now but the first line I type into python interactive editor ( `import tensorflow as tf` ) gets nasty reply:
> `/Users/tensorflow/anaconda3/envs/tensorflow/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6`

I see that ppp1d has python 3.6 in ~/anaconda3/usr/bin/ so where did this 3.5 module of fast_tensor_util come from?? Maybe I need a reboot or a re-install?


## Questions
* Is it necessary to install homebrew in order to install Anaconda? I'm trying Anaconda w/o homebrew in ppp1d. An [article in stackoverflow](https://stackoverflow.com/questions/33541876/os-x-deciding-between-anaconda-and-homebrew-python-environments) says anaconda and homebrew install separate python in different places.
* Anaconda 5.0 can install tensorflow for us, but how sketchy is the "community" support of this install vs the "official" install of tensorflow? tensorflow.org has [page about using anaconda to install tensorflow](https://www.tensorflow.org/install/install_mac#installing_with_anaconda)
* How do we update Anaconda? (Anaconda-Navigator self checks for updates.)
* Does the GitHub setting about hide email really hide email address from commits done by desktop GitHub? If using command line, see [email in git](https://help.github.com/articles/setting-your-email-in-git) about adjusting GitHub to hide email in command line activity.


## Questions that we've answered
* How to install pip: Anaconda 5.0 installed python 3.6 including pip.
* How to install numpy: Anaconda 5.0 installed numpy.


## Done (In Order!)
* (√ Mar2018) Using [homebrew](http://brew.sh) to start [installing gym.openai.com](github.com/openai/gym#basics) (began with `brew update` and then `brew install`
* (√ Oct2017) Perhaps was unnecessary? Installed [homebrew](http://brew.sh) (required a sudo for installation, required that our computer account be an administrator (so much for student installs) "This script requires the user tensorflow to be an Administrator.", and says that package downloads will NOT need sudo).
* (√ Oct2017) Installed Anaconda (which created ~/anaconda3 and did not need an admin password, offered to install python 3.6 and/or 2.7 (we installed 3.6)).
* (√ Oct2017) Used Anaconda to install tensorFlow (after reviewing ownership of "/usr/local/Cellar"  using

   `su adminAcct`
   
   `sudo chown -R tensorflow /usr/local/Cellar`
   
   `sudo chown -R tensorflow /usr/local/var/homebrew`
   
   `sudo chown -R tensorflow /usr/local/Homebrew`
   
   `sudo chown -R tensorflow /usr/local/opt`
   
   ?? Why  /usr/local/bin priv issue with gym install??
 
 -- note: homebrew's suggestion was 
 
   `sudo chown -R $(whoami) /usr/local/Cellar`
   
Note: “whoami” displays effective user id but man file says “has been obsoleted by the ‘id’ utility and is equivalent to ‘id -un’ though ‘id -p’ is suggested for normal interactive use.”
Note: We're seeing that Anaconda's install of tensorflow puts lots of invisible (dot files) stuff in "~" home folder. Would have been nice if it made a single folder for all the tensorflow stuff.

* (√ Oct2017) Told Mac Finder to show our invisible files:
adjusting the 3rd word of the following to match your account name (example account is called 'tense')

   `defaults -host` **tense** `write com.apple.finder AppleShowAllFiles -bool YES`
   
Restart finder after doing this: can terminal `killall Finder` or forceQuit finder or option-click finder in dock and choose "Relaunch"
