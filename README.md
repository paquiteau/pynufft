![](g5738.jpeg)
# PyNUFFT: Python non-uniform fast Fourier transform

A minimal "getting start" tutorial is available at http://jyhmiinlin.github.io/pynufft/ .

## How not to misuse PyNUFFT?

The best way is either to use it correctly or not to use this package at all. 

We release the package in the hope that it is useful. However, some users started to abuse this service and make false claims.

We kindly ask users stop spreading false information: 
when they only have data saved as single precision floating point numbers, they assigned wrong parameters and claim that PyNUFFT is worse than other packages. (which is not true) 
 
For example, some users have set the Cartesian grid incorrectly, and claim that PyNUFFT is far worse than they expected. What made their example failed is that the coordinate was wrongly set. We have made the correction in another conference. The SSIM of the correct Cartesian setting is SSIM ~ 1.000. However, we don't have time to make corrections for all cases. 
 
We have collected many failed case studies (preprints, emails from big organizations, tutorials in signal processing conferences). 

We are offering a training course to allow users to access these failed cases and learn how to use the package correctly.


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

![](speed_accuracy_comparisons.png)

### Contact information
email: pynufft@gamil.com
