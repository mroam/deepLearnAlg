Course: https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python/

Notes
--------
Lecture 1: Starts with NumPy arrays (just like that article from medium)

* Central object is "DataFrame"
* gonna use matplotlib to look at data
* "scipy" adds stats functions to numpy

Lecture 2: 
* Suggests using [ipython](https://ipython.org/) (which runs interactive python inside the [jupyter](https://jupyter.org/) shell and notebook system). Anaconda installed iPython for us and provides console.
* Suggests following command (preceded with sudo, but is sudo required??) for installing necessaries onto mac:
pip install -U numpy scipy matplotlib pandas ipython
We might not need this since we've used Anaconda to install stuff including tensorFlow.


Questions: 
--------
* What (in stats) is "Convolution"? Well, it is a function that somehow combines two other functions to get a new function. Check out https://en.wikipedia.org/wiki/Convolution


Pre-Req Knowledge to review
-------
* "Gaussian Distributions" in 1-d and 2-d
* Python
* Vector and matrix operations. 

Notes re Matrix and Vector operations: 
* [ad,be,cf] = [a,b,c] * [d,e,f]
* dot product aka "scalar product" is called in numpy by np.dot(A,B) (also A.dot(B) or B.dot(A) and works ad+be+cf = [a,b,c] dot [d,e,f]
* cross product aka "vector product" is np.cross(A,B) and is often written "x" and finds perpendicular vector.
* "matrix product" is the kind of multiplication of matrices used in computer graphics, which is neither "dot" nor "cross".


Handy Commands
--------
A = np.array([1,2,3])
B = np.array([4,5,6])
C = np.random.random(5)     // gives 5 element vector of values between 0..1
print(A)

Resources:
--------
* Class materials for download: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/numpy_class
