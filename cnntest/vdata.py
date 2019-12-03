
# https://en.wikipedia.org/wiki/Kernel_(image_processing)
# https://docs.scipy.org/doc/scipy-1.0.0/reference/tutorial/signal.html#convolution-correlation
# https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb#scrollTo=jRFxccghyMVo

x = np.zeros((4096, 4))
x[:,0] = norm_data.BR_norm[0:4096]
x[:,1] = norm_data.BTH_norm[0:4096]
x[:,2] = norm_data.BPH_norm[0:4096]
x[:,3] = norm_data.BMAG_norm[0:4096]
h = np.array([[1,2,1],[2,4,2],[1,2,1]])
h = np.array([1,1,1])
z = scipy.signal.convolve(x[:,0], h)

ii = np.arange(0,4096,1)
pl.plot(ii, norm_data.BR_norm[0:4096])
pl.plot(ii, z[1:4097])
pl.show()
