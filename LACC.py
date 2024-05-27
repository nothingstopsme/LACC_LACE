import cv2
import numpy as np

def process(I, stop_criterion=1e-2, noise_kernel_size=25, noise_std=1.0):
  '''
  I: the input image of which the channel format is normalsed RGB or normalised BGR, i.e. the range of channel values is [0.0, 1.0]

  stop_criterion: when the loss being minimised is less than or equal to this value, stop the iterative optimisation loop

  noise_kernel_size: the size of the square gaussian kernel for blurring the input I

  noise_std: the standard deviation of the 2d isotropic gaussian distribution used for blurring the input I

  '''

  mean = np.mean(I, axis=(0, 1))
  order_indices = np.argsort(mean)
  initial_mean_s = mean_s = mean[order_indices[0]]
  initial_mean_m = mean_m = mean[order_indices[1]]
  initial_mean_l = mean_l = mean[order_indices[2]]
 
  I_copy = I.copy()
  # I_s/I_m/I_l are views of I_copy
  I_s = I_copy[..., order_indices[0]]
  I_m = I_copy[..., order_indices[1]]
  I_l = I_copy[..., order_indices[2]]

  min_I_l = np.min(I_l)
  max_I_l = np.max(I_l)
  I_l = (I_l - min_I_l) / (max_I_l - min_I_l)
  mean_l = np.mean(I_l)

  # Minimising the loss function loss1 + loss2 by iteratively
  # update I_s/mean_s and I_m/mean_m based on I_l/mean_l
  while True:

    loss1 = np.abs(mean_l - mean_m)
    loss2 = np.abs(mean_l - mean_s)
    #print(f'|loss| = {loss1 + loss2}')
    if loss1+loss2 <= stop_criterion:
      break
    

    I_s += (mean_l - mean_s) * I_l
    I_m += (mean_l - mean_m) * I_l
    mean_s = np.mean(I_s)
    mean_m = np.mean(I_m)

  

  I_ct_list = [(I_s, order_indices[0]), (I_m, order_indices[1]), (I_l, order_indices[2])]
  I_ct_list.sort(key=lambda pair: pair[1])
  I_ct = np.stack([I_ct_channel for I_ct_channel, _ in I_ct_list], axis=-1)

  D = I - cv2.GaussianBlur(I, (noise_kernel_size, noise_kernel_size), noise_std)
  one_minus_AM = I[..., order_indices[0]:order_indices[0]+1]**1.2
  #one_minus_AM = np.min(I, axis=-1, keepdims=True)**1.2


  return np.clip(D + (1.0 - one_minus_AM) * I_ct + one_minus_AM * I, 0.0, 1.0)
  

