import cv2
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve
from google.colab import files

def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map
  
def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(c - new_c):
        img = carve_column(img)

    return img
  
def obj_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(c - new_c):
        img = obj(img)

    return img

def inc_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(new_c -c):
        img = increase_column(img)

    return img
  
def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def inc_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = inc_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img
  
def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def obj(img):
    r, c, _ = img.shape

    M, backtrack = maximum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmax(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def increase_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    new = np.ones((r, c + 1, 3))
    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        new[i,:j+1,1]=img[i,:j+1,1]
        new[i,j+1,1]=img[i,j,1]
        new[i,j+2:c+1,1]=img[i,j+1:c,1]
        new[i,:j+1,2]=img[i,:j+1,2]
        new[i,j+1,2]=img[i,j,2]
        new[i,j+2:c+1,2]=img[i,j+1:c,2]        
        new[i,:j+1,0]=img[i,:j+1,0]
        new[i,j+1,0]=img[i,j,0]
        new[i,j+2:c+1,0]=img[i,j+1:c,0]
        j = backtrack[i, j]
      
    return new
  
def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)
    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
#                 print(idx+j)
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]
#                 print(idx+j-1)
            M[i, j] += min_energy

    return M, backtrack

def maximum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)
    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmax(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                max_energy = M[i-1, idx + j]
#                 print(idx+j)
            else:
                idx = np.argmax(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                max_energy = M[i - 1, idx + j - 1]
#                 print(idx+j-1)
            M[i, j] += max_energy

    return M, backtrack
  
def main():
  imgg = cv2.imread("hiking-people-on-snowy-mountains-4864x2736_98684.jpg")
  img_cvt=cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
  plt.imshow(img_cvt)
  plt.title('Input Image'),plt.xticks([]),plt.yticks([])
  plt.show()
  r, c, _ = img_cvt.shape
  ch=int(input("1. Decrease Width\n2. Decrease Height\n3. Remove Object\n4. Increase Width\n5. Increase Height\n"))
  if ch==1:
      scale_c=float(input("Enter Value of Scale for column(<1) "))
      out=crop_c(img_cvt, scale_c)
  elif ch==2:
      scale_r=float(input("Enter Value of Scale for row(<1) "))
      out=crop_r(img_cvt, scale_r)
  elif ch==3:
      out=obj_c(img_cvt, 0.7)
  elif ch==4:
      scale_c=float(input("Enter Value of Scale for column(>1) "))
      out=inc_c(img_cvt, scale_c)
  elif ch==5:
      scale_r=float(input("Enter Value of Scale for row(>1) "))
      out=inc_r(img_cvt, scale_r)
      
  a,b,cc=out.shape
  print(str(a)+' '+str(b)+' '+str(cc))
#   color = np.ones((r, c+1))#, dtype=np.uint8)
#   color[:,:]=out[:,:,0]+out[:,:,1]+out[:,:,2]
#   print(img_cvt[:,:,0])
#   print(out[:,:,0].astype(int))
#   out[:,:,0]=out[:,:,0].astype(int)
#   out[:,:,1]=out[:,:,1].astype(int)
#   out[:,:,2]=out[:,:,2].astype(int)
#   o=cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
#   o=cv2.cvtColor(color,cv2.COLOR_GRAY2RGB)
#   out_cvt=cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
#   scaley=float(input("Enter Value of Scale for row "))       #0.988
#   final = crop_r(out, scaley)
  plt.imshow(out)
  plt.title('Output Image'),plt.xticks([]),plt.yticks([])
  plt.show()
  ro, co, _ = out.shape
#   rf, cf, _ = final.shape
  print('original dimensions- '+str(r)+' '+str(c))
  print('after seam carving in column dimensions- '+str(ro)+' '+str(co))
#   with open('example.jpg', 'w') as f:
#     f.write('out')
#   print('after seam carving dimensions- '+str(rf)+' '+str(cf))
if __name__ == '__main__':
    main()
