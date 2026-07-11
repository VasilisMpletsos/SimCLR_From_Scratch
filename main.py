from torch.nn import Module
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %% Test 1
random_y = np.asarray([0,1,2,3,4,5])
random_x = np.asarray([2,3,5,7,15,30])

# %% Test 2
sns.lineplot(x=random_x, y=random_y)
plt.title("Log")
plt.title("Test2")
