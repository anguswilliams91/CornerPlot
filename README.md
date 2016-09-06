# CornerPlot
A corner plotting routine for MCMC output in `python`. Requires `numpy` and `matplotlib`. 

To install using pip, run

`pip install https://github.com/anguswilliams91/CornerPlot/archive/master.zip`

which will also install missing dependencies. The module is called `corner_plot`. The plotting 
function is also called `corner_plot` and has an informative docstring. Below is an example output, 
where samples from a unit, isotropic Gaussian are plotted.

```python
import corner_plot as cp
import numpy as np

#draw samples from a unit, isotropic Gaussian
samples = np.random.multivariate_normal([0.,0.,0.],[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],size=1000000)

#plot
cp.corner_plot(samples)

```

![Alt text](example.png?raw=true)
![Alt text](example_1.png?raw=true)

