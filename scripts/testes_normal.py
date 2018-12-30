from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt

dados = norm.rvs(size = 100)
stats.probplot(dados, plot = plt)

stats.shapiro(dados)

import pandas as pd
import numpy as np
a = pd.DataFrame(np.arange(10)*10)
a

