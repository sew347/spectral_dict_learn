import numpy as np

#return the orthogonal complement of x to the subspace spanned by SSP
def subspace_proj(x,SSP):
	resid = x
	dims = np.shape(SSP)
	for i in range(dims[1]):
		resid = resid - np.inner(resid,SSP[:,i])*SSP[:,i]
	proj = x - resid
	return proj, resid
