# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    def fexmatch(i1,i2):
        sift = cv2.xfeatures2d.SIFT_create()
        kp1 = sift.detect(i1, None)
        kp1,descp1 = sift.compute(i1,kp1)
        kp2 = sift.detect(i2,None)
        kp2,descp2 = sift.compute(i2,kp2)
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
        return c1,c2,len(gooddescp),len(descp1),i2

    gimg=[]
    for gi in imgs:
        temp = cv2.cvtColor(gi,cv2.COLOR_BGR2GRAY)
        gimg.append(temp)
    
    num=np.zeros((4,4))
    olist=np.zeros((4,4))
    for i in range(len(gimg)):
        for j in range(len(gimg)):
            p1,p2,lgdes,ldes1,tempim = fexmatch(gimg[i],gimg[j])
            overlapflag=lgdes/ldes1*1000
            print(overlapflag)
            if overlapflag>=0.2:    #If more than 20%
                num[i,j]=lgdes
                olist[i,j]=1
    print("Overlap array is:")
    print(olist)
    x=np.arange(len(imgs)) 
    for i in range(len(imgs)):
        x[i] = np.count_nonzero(num[i,:])
    max = np.max(x)
    maxin = np.asarray(np.where(x == max))
    order=list()
    sum=np.arange(len(maxin))
    for i in range(len(maxin)):
        sum[i]=np.sum(num[maxin[i],:]) - num[maxin[i],maxin[i]]
        
    maxs=np.max(sum)
    in1 = np.asarray(np.where(sum == maxs)).reshape(-1)
    order.append(np.asscalar(maxin[0,in1]))
    sq = num[maxin[0,in1],:]
    sortreverse = np.sort(sq)[::-1]
    sortreverse=sortreverse[1:len(sortreverse)]
    for i in range(len(sortreverse)):
      l = np.asarray(np.where(sq == sortreverse[i])).flatten()
      order.append(np.asscalar(l))
      
    fim = imgs[order[0]]
    for i in range(1,len(order)):
        pt1,pt2,lgd,ld1,i2= fexmatch(fim, imgs[order[i]])
        h, status = cv2.findHomography(pt1, pt2, cv2.RANSAC, 5.0)
        #padimg1 = cv2.copyMakeBorder(i1,100,100,100,100,cv2.BORDER_CONSTANT,value=[0,0,0])
        wimg = cv2.warpPerspective(fim, h, ((i2.shape[0]+i2.shape[1]),(fim.shape[1]+i2.shape[1])))

        plt.imshow(wimg)
        plt.show()
        temp1=np.zeros((wimg.shape))
        temp2=np.zeros((wimg.shape))
        temp1[0:wimg.shape[0],0:wimg.shape[1]] = wimg
        temp2[0:i2.shape[0], 0:i2.shape[1]] = i2
        wimg[0:i2.shape[0], 0:i2.shape[1]] = i2 #stitched image

    return olist
if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
# =============================================================================
#     #bonus
#     overlap_arr2 = stitch('t3', savepath='task3.png')
#     with open('t3_overlap.txt', 'w') as outfile:
#         json.dump(overlap_arr2.tolist(), outfile)
# 
# =============================================================================
