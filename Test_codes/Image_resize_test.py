import PIL.Image as Image
import numpy as np

img1 = Image.fromarray(255*np.array([[0.,1.],[1.,0.]]))
img2 = img1.resize([100,100])
img1 = img1.convert(mode='L')
img2 = img2.convert(mode='L')
img1.save('test1.bmp')
img2.save('test2.bmp')
