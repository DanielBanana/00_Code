import numpy as np




p = np.asarray([[1,2,3],[2,4,6]])

b_c = np.asarray([[[0,1,11]],[[0.2,2,12]],[[0.4,3,13]],[[0.6,4,14]],[[0.8,5,15]]])

n_particles = p.shape[0]
n_samples = b_c.shape[0]
n_outputs = b_c.shape[1]
n_parameters = b_c.shape[2]



res = np.empty((n_particles, n_samples, n_outputs, n_parameters))

for x in range(n_particles):
    for b in range(n_samples):
        for o in range(n_outputs):
            res[x, b, o, :] = p[x, :] - b_c[b, o, :]



# res = np.einsum('xp,bop->xbop',p, -b_c)
print(res)