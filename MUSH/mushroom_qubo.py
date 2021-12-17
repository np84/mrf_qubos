import numpy as np
import pxpy as px


D,M = px.tools.genfromstrcsv('mushroom_no_missing.csv')

m = px.train(D, graph=px.GraphType.auto_tree, iters=1000)

x = m.MAP[0]

print('MAP = ',end='')
for i in range(len(x)):
	print(M[i][x[i]], end='')
print()

Q = m.export_qubo()
np.save('mushroom.npy',Q)
