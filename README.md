![](g5738.jpeg)
# PyNUFFT: Python non-uniform fast Fourier transform

A minimal "getting start" tutorial is available at http://jyhmiinlin.github.io/pynufft/ .
 
## Installation

$ pip install pynufft --user


### Using Numpy/Scipy

```
$ python
Python 3.6.11 (default, Aug 23 2020, 18:05:39) 
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from pynufft import NUFFT
>>> import numpy
>>> A = NUFFT()
>>> om = numpy.random.randn(10,2)
>>> Nd = (64,64)
>>> Kd = (128,128)
>>> Jd = (6,6)
>>> A.plan(om, Nd, Kd, Jd)
0
>>> x=numpy.random.randn(*Nd)
>>> y = A.forward(x)
```

### Using PyCUDA

```
>>> from pynufft import NUFFT, helper
>>> import numpy
>>> A2= NUFFT(helper.device_list()[0])
>>> A2.device
<reikna.cluda.cuda.Device object at 0x7f9ad99923b0>
>>> om = numpy.random.randn(10,2)
>>> Nd = (64,64)
>>> Kd = (128,128)
>>> Jd = (6,6)
>>> A2.plan(om, Nd, Kd, Jd)
0
>>> x=numpy.random.randn(*Nd)
>>> y = A2.forward(x)
```

## Testing GPU acceleration

```
Python 3.6.11 (default, Aug 23 2020, 18:05:39) 
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from pynufft import tests
>>> tests.test_init(0)
device name =  <reikna.cluda.cuda.Device object at 0x7f41d4098688>
0.06576069355010987
0.006289639472961426
error gx2= 2.0638987e-07
error gy= 1.0912560261408778e-07
acceleration= 10.455399523742015
17.97926664352417 2.710083246231079
acceleration in solver= 6.634211944790991
```
## Abuse of the project

We have noticed many people have started to abuse this project as soon as 
they know this work. 

They were using their junky benchmarks, and use propaganda to delibrately lower 
the performance of PyNUFFT. 

As we have shown in the tutorial, their misunderstanding about Fourier transform
is the reason why they failed. 
'''
https://github.com/philouc/isbi19-tutorial/blob/f442a89c4fc8c76fa225bedbf6804c6fa300e56c/python/03b.ISBI19_notebook.ipynb
'''
As they have violated the open-source license, 
their right to continuously use this package has been revoked. 

### Contact information
email: pynufft@gamil.com
