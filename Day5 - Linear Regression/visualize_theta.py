import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("Training Data/Linear_X_Train.csv").values
Y = pd.read_csv("Training Data/Linear_Y_Train.csv").values


thetas = np.load("thetas.npy")

T1 = thetas[:, 1]
T0 = thetas[:, 0]


plt.ion()
for i in range(1, 50, 3):
	Y_pred = T0[i] + T1[i]*X

	plt.scatter(X, Y)
	plt.plot(X, Y_pred, color="red")
	plt.draw()
	plt.pause(1)
	plt.clf()