"""read and display data from csv at url
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pandas as pd

#csvname = 'http://cvl.inf.tu-dresden.de/HTML/teaching/courses/PML2/ws18/Data_3D_2classes.csv'
csvname = './Data_3D_2classes.csv'

print('Reading "' + csvname + '":')
dat = np.loadtxt(csvname, delimiter=';')

# print(dat)

df = pd.DataFrame(dat)
print(df.describe())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, d in enumerate(dat):
    if d[3] > 0:
        ax.scatter(d[0], d[1], d[2], color='#0000FF', marker='o')
    else:
        ax.scatter(d[0], d[1], d[2], color='#FF8000', marker='o')
                   
plt.title('Distribution (two classes)')
ax.set_xlabel('Random points')
ax.set_ylabel('Class 1 = Blue, Class -1 = Orange')
ax.set_zlabel('Z Label')
plt.show()
