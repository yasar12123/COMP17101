import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.utils import to_categorical

label = [0,1,2,0,1,0,2,2]
label = to_categorical(label, 3)
print(label)
label = np.argmax(label, axis=-1)

print(label)