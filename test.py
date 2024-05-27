import os
import cv2
import numpy as np

import LACC
import LACE

def main():
  input_root_path = "input"
  output_root_path = "output"

  os.makedirs(output_root_path, exist_ok=True)
  with os.scandir(input_root_path) as entries:
    for entry in entries:
      if entry.is_file():
        # Assuming all files are images to be processed 
        img = cv2.imread(entry.path)
        print(f'Processing {entry.path} {img.shape} ...', end='', flush=True)

        img = img.astype(np.float32)
        ct_img = LACC.process(img / 255.0)
        Lab = cv2.cvtColor(ct_img, cv2.COLOR_BGR2Lab)
        enhanced_Lab = LACE.process(Lab)

        enhanced_BGR = cv2.cvtColor(enhanced_Lab, cv2.COLOR_Lab2BGR)

        output_path = os.path.join(output_root_path, os.path.relpath(entry.path, start=input_root_path))
        cv2.imwrite(output_path, np.round(enhanced_BGR * 255.0).astype(np.uint8))
        print('done')


if __name__ == '__main__':
  main()

