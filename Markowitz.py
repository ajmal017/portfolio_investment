#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
import plotly
import cufflinks
# (*) To communicate with Plotly's server, sign in with credentials file
import chart_studio.plotly as py
# (*) Useful Python/Plotly tools
import chart_studio.tools as tls
# (*) Graph objects to piece together plots
from chart_studio.graph_objs import *

np.random.seed(123)

# Turn off progress printing
solvers.options['show_progress'] = False

## NUMBER OF ASSETS
n_assets = 4

## NUMBER OF OBSERVATIONS
n_obs = 1000

return_vec = np.random.randn(n_assets, n_obs)

fig = plt.figure()
plt.plot(return_vec.T, alpha=.4);
plt.xlabel('time')
plt.ylabel('returns')
py.iplot_mpl(fig, filename='s6_damped_oscillation')

