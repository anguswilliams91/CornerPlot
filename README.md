# CornerPlot
A corner plotting routine for MCMC output in `python`. Requires `numpy` and `matplotlib`. 

If you already have `numpy` and `matplotlib`, then install the package using `pip` by 
running

`pip install https://github.com/anguswilliams91/CornerPlot/archive/master.zip`

The plotting function is also called `corner_plot` and has an informative docstring. 
Below is an example output, where samples from a unit, isotropic Gaussian are plotted.

```python
import corner_plot as cp
import numpy as np

#draw samples from a unit, isotropic Gaussian
samples = np.random.multivariate_normal([10.,5.,100.],[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],size=1000000)

#plot
axis_labels=["$x$","$y$","$z$"]
cp.corner_plot(samples,axis_labels=axis_labels)
cp.corner_plot(samples,gradient=True,linewidth=0.,nbins=100,axis_labels=axis_labels)
cp.corner_plot(samples[::1000],scatter=True,filled=False,scatter_size=2,axis_labels=axis_labels,nbins=10)


```

If you have constrained the same model with different data sets, you can now plot the inference from each 
data set on the same corner plot to compare the results. This is implemented in a function `multi_corner_plot`. 
Example:

```python

import corner_plot as cp
import numpy as np

samples = np.random.multivariate_normal([10.,21.,101.],[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],size=1000000)
samples1 = np.random.multivariate_normal([12.,20.,100.],[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],size=1000000)
samples2 = np.random.multivariate_normal([9.,22.,98.],[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],size=1000000)

chains = (samples,samples1,samples2)

cp.multi_corner_plot(chains,axis_labels=["$x$","$y$","$z$"],linewidth=2.,\
                                            chain_labels=["data 1","data 2","data 3"])

```
![example](https://cloud.githubusercontent.com/assets/6830677/21053340/e84bade8-be20-11e6-9eed-58cdc4e3ad2f.png)
![example_1](https://cloud.githubusercontent.com/assets/6830677/21053349/f16a7c42-be20-11e6-9b65-8c62c696ac2f.png)
![example_2](https://cloud.githubusercontent.com/assets/6830677/21053354/f582cf6e-be20-11e6-927b-55805f3f22b1.png)
![example_3](https://cloud.githubusercontent.com/assets/6830677/21053362/fb482d04-be20-11e6-95b1-b2e2be46529b.png)

