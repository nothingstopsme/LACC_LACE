import cv2
import numpy as np

def process(Lab, block_size=25, epsilon=1e-2):
  '''
  Lab:
  The input image in the CIEL*a*b* format.

  block_size:
  The size of square blocks for calculating local statistics;
  this is also the size of the square kernel used for guilded image filtering.

  epsilon:
  The parameter of guilded image filtering which controls the degree of local variances being preserved.
  Its value should be >= 0.0, and the larger it is, the stronger smoothing effect is produced.
  '''

  L = Lab[..., 0]
  ab = Lab[..., 1:]

  ############## for L ##############

  kernel = np.ones((block_size, block_size))
  normalised_kernel = kernel / (block_size*block_size)
  centre = (block_size+1) // 2 - 1

  # The anchor of the kernel
  anchor = np.array((centre, centre))
  
  # the anchor of the flipped version (both horizontally and vertically) of the kernel. For a w by w square kernel with an odd w,
  # the anchors for both the non-flipped and flipped kernel are the same
  anchor_after_flipped = block_size - anchor - 1
  
  block_mean = cv2.filter2D(L, -1, normalised_kernel, anchor=anchor)
  
  # Making sure block variances are at least 1e-8,
  # such that the division operation later which takes variances as denominator will not cause 'divided-by-zero' errors.
  #
  # Note that ratios computed from too small block variances ends up being capped at 2.0,
  # so placing the 1e-8 lower bound is highly unlikely to change the supposed ratio outcomes
  block_var = np.maximum(cv2.filter2D(L**2, -1, normalised_kernel, anchor=anchor) - block_mean**2, 1e-8)
  global_var = np.var(L)
  ratio =  np.minimum(global_var / block_var, 2.0)
  
  # for each pixel, the scale and offset are averaged over all blocks in which that pixel is involved,
  # so its value in L_eb is equivalent to the average of all mapped pixel values computed from different blocks

  #L_eb = block_mean + ratio * (L - block_mean)
  L_eb = cv2.filter2D(ratio, -1, normalised_kernel, anchor=anchor_after_flipped) * L + cv2.filter2D((1.0 - ratio) * block_mean, -1, normalised_kernel, anchor=anchor_after_flipped)

  L_min = cv2.erode(L_eb, kernel, anchor=anchor, borderType=cv2.BORDER_DEFAULT)
  L_max = cv2.dilate(L_eb, kernel, anchor=anchor, borderType=cv2.BORDER_DEFAULT)

  max_minus_min = L_max - L_min
  
  L_eb_mean = cv2.filter2D(L_eb, -1, normalised_kernel, anchor=anchor)
  L_eb_squared_mean = cv2.filter2D(L_eb**2, -1, normalised_kernel, anchor=anchor)
  L_eb_var = np.maximum(L_eb_squared_mean - L_eb_mean**2, 0.0)

  # While the paper does not include the epsilon term in its "guided image filtering" equation, it must be there as without it, 
  # k will degrade to max_minus_min and v will degrade to L_min,
  # which implies the output of the guided image filtering is just the original input L_eb, and thus no filtering effect at all.
  k_over_max_minus_min = L_eb_var / (L_eb_var + max_minus_min**2 * epsilon)

  # It is k_over_max_minus_min, not k, that is actually used in the computation below
  #k = max_minus_min * k_over_max_minus_min
  v = L_eb_mean - k_over_max_minus_min * (L_eb_mean - L_min)

  
  # For each block, since guide(i, j) = (L_eb(i, j) - L_min) / max_minus_min,
  # => k * guide(i, j) + v = k_over_max_minus_min * L_eb(i, j) + (v - k_over_max_minus_min * L_min,
  # and therefore the cv2.filter2D() calls below are invoked to average those values over blocks
  L = cv2.filter2D(k_over_max_minus_min, -1, normalised_kernel, anchor=anchor_after_flipped) * L_eb + cv2.filter2D(v - k_over_max_minus_min * L_min, -1, normalised_kernel, anchor=anchor_after_flipped)
  L = L[..., np.newaxis]  


  ############## for ab ##############

  # Temperarily shifting the range of values for both a* and b* channel to [1, 255],
  # so that the denominator, which is defined as mean_a + mean_b, will not be 0
  ab += 128.0
  mean_ab = np.mean(ab, axis=(0, 1))
  
  denominator = mean_ab.sum()

  if mean_ab[0] < mean_ab[1]:
    factor = np.stack(((mean_ab[1] - mean_ab[0]) / denominator, 0.0))
  elif mean_ab[0] > mean_ab[1]:
    factor = np.stack(((0.0, mean_ab[0] - mean_ab[1]) / denominator))
  else:
    factor = np.zeros(2)

  # Applying the compensation and inverting the shifted range back to the original one
  ab += factor * ab - 128.0

  return np.concatenate((np.clip(L, 0, 100), np.clip(ab, -127, 127)), axis=-1)

