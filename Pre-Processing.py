import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

output_dir='Pre_imgs\\'
for img_dir in glob.glob('E:\\data\\BUE\\Year 3\\Semster one\\GP\\Nour\\Main\\Pre_processing\\Data_set\\*.png'):
    img = cv2.imread(img_dir, 0)
    image = cv2.resize(img, (224, 224))
    cv2.imshow("equalizeHist", image)
    w = cv2.equalizeHist(image)
    kernel1 = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(w, -1, kernel1) # applying the sharpening kernel to the input image & displaying it.
    res = np.hstack((image, w,sharpened))
    cv2.imshow("equalizeHist", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_dir+img_dir.split('\\')[-1],sharpened)

