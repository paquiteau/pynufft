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

### Using NUDFT_cupy and NUDFT (double precision, experimental)

Some users ask for double precision. 
NUDFT and NUDFT_cupy are offered.
Speedup is dependent on cupy and GPU.  


```
>>> from pynufft import NUDFT_cupy, NUDFT
>>> import numpy
>>> A2= NUDFT_cupy()
>>> om = numpy.random.randn(10,2)
>>> Nd = (64,64)
>>> A2.plan(om, Nd)
>>> x=numpy.random.randn(*Nd)
>>> y = A2.forward(x)
>>> A = NUDFT()
>>> A.plan(om, Nd)
>>> y_cpu = A.forward(x)
>>> print(numpy.linalg.norm(y.get() - y_cpu))
6.752054788357788e-14
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

![](Figure_1.png)

### On the wrong information in the IEEE-ISBI 2019 tutorial

Here is the fork that I took at that time. 

https://github.com/jyhmiinlin/isbi19-tutorial

I don't know why they are doing this?


There are just too many errors in that tutorial. 
PyNUFFT aims to provide a fast and accurate NUFFT implementation on OpenCL/CUDA devices, as well as Numpy/Scipy. 
There is a condition that users of the PyNUFFT package should develop their solver, instead of delivering wrong information and abuse this package.
This action is an abuse of this package.

In addition, their application is built on top of PyNUFFT, which I think their approach is deliberately shedding a negative light on PyNUFFT. 

There are too many obvious errors in the Jupyter Notebook given in IEEE-ISBI 2019 tutorial.                                                 
One of our colleagues has corrected their error in ESMRMB 2019 but there are more errors so we don't want to spend more time on this material. 
We have asked them to correct their error in their tutorial, but I have received no correction. 
We can only educate users who would like to listen.
 

### On the Off-the-grid data-driven optimization of sampling schemes...

https://arxiv.org/pdf/2010.01817

" ...... and Python toolboxes begin to emerge [cite PyNUFFT]. Our experience using them however led to unstable results due to significant numerical errors."
which is wrong. If they want double-precision, they can use NUDFT and NUDFT_cupy. I have kindly sent an email them to correct their statement.  


### Contact information
J.-M. Lin
email: pynufft@gamil.com

