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
![default](https://cloud.githubusercontent.com/assets/6830677/21099547/b64998a8-c066-11e6-8f67-f2a71d960bad.png)
![gradient](https://cloud.githubusercontent.com/assets/6830677/21099550/ba42d17c-c066-11e6-875c-60e295be0890.png)
![scatter](https://cloud.githubusercontent.com/assets/6830677/21099553/bdfe71a4-c066-11e6-8797-83ac1010cc84.png)
![multi](https://cloud.githubusercontent.com/assets/6830677/21099555/c0ab494a-c066-11e6-880f-b0656e55f0bd.png)



