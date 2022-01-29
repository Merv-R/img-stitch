#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    gimg1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gimg2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    plt.imshow(gimg1)
    plt.show()
    plt.imshow(gimg2)
    plt.show()
    # sift = cv2.SIFT_create()
    sift = cv2.xfeatures2d.SIFT_create()
    kp1 = sift.detect(gimg1,None)
    #img=cv2.drawKeypoints(gimg1,kp1,img1)
    kp1,descp1 = sift.compute(gimg1,kp1)
    #kpts1 = np.float32([keypt.pt for keypt in kp1])
    kp2 = sift.detect(gimg2,None)
    #img=cv2.drawKeypoints(gimg2,kp2,img2)
    kp2,descp2 = sift.compute(gimg2,kp2)
    #kpts2 = np.float32([keypt.pt for keypt in kp2])
    
    """
    Reference for my KNN Algorithm: 
        https://www.cs.cornell.edu/courses/cs1114/2013sp/sections/S10_sift.pdf
    """
    
    gooddescp=[]
    for i in range(len(kp1)):
        d1=np.array(descp1[i,:])
        distance=[]
        for j in range(len(kp2)):
            d2=np.array(descp2[j,:])
            d=np.subtract(d2,d1)
            dt = np.transpose(d)
            prod = np.matmul(d,dt)
            dist = np.sqrt(prod)
            distance.append(dist)
        sorteddist = np.sort(distance)
        min1=sorteddist[0] #min value
        min2=sorteddist[1] #second min value
        if(min1 < 0.6 * min2):
            k=np.asarray(np.where(distance==min1))
            val = [i,k.item()]
            gooddescp.append(val)
        
# =============================================================================
#     goodmatches = np.asarray(gooddescp)
#     src = []
#     for i in goodmatches:
#         temp = np.float32(kp1[i].pt)
#         src.append(temp)
#     dest = []
#     for j in goodmatches:
#         temp = np.float32(kp2[j].pt)
#         dest.append(temp)
# =============================================================================
    """
    Reference for Homography usage: 
        https://www.pythonpool.com/cv2-findhomography/
        https://www.programcreek.com/python/example/89367/cv2.findHomography
    """
    
    coords1=[]
    coords2=[]
    for i in range(len(gooddescp)):
       coords1.append(kp1[(gooddescp[i][0])].pt)
       coords2.append(kp2[(gooddescp[i][1])].pt)
       
    c1 = np.float32(coords1).reshape(-1,1,2)
    c2 = np.float32(coords2).reshape(-1,1,2)        
    h, status = cv2.findHomography(c1, c2, cv2.RANSAC, 5.0)
    #padimg1 = cv2.copyMakeBorder(i1,100,100,100,100,cv2.BORDER_CONSTANT,value=[0,0,0])
    wimg = cv2.warpPerspective(img1, h, ((img2.shape[0]+img2.shape[1]),(img1.shape[1]+img2.shape[1])))
    plt.imshow(img1)
    plt.show()
    plt.imshow(wimg)
    plt.show()
    temp1=np.zeros((wimg.shape))
    temp2=np.zeros((wimg.shape))
    temp1[0:wimg.shape[0],0:wimg.shape[1]] = wimg
    temp2[0:img2.shape[0], 0:img2.shape[1]] = img2
    wimg[0:img2.shape[0], 0:img2.shape[1]] = img2 # primitive stitching
    plt.imshow(temp1)
    plt.show()
    plt.imshow(temp2)
    plt.show()
    plt.imshow(wimg)
    plt.show()
    xorimg = cv2.bitwise_xor(temp1, temp2)
    plt.imshow(xorimg)
    plt.show()
    cv2.imwrite(savepath, wimg)

    return

if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

