import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time as t
import sys
script = sys.argv[0]
path = sys.argv[1]
file = open("/home/dikshit/Desktop/corpusHist.txt","r")
c = file.read(1)
x = 0
while c!='\n':
    x = x*10 +(ord(c)-48)
    c= file.read(1)
n = x
start = t.time()
hist = np.zeros([n,30])
i=0
j=0
x = 0
while True:
    c = file.read(1)
    if not c:
        break
    if c == '\n':
        i = i + 1
        j = 0
    elif c!=' ':
        x = x*10 + (ord(c) - 48)
        hist[i][j] =  x
    elif c == ' ':
        j = j + 1
        x = 0
file.close()
import numpy as np
import math
cent = np.zeros([42,32])
f = open("/home/dikshit/Desktop/centroid.txt","r")
i=0
j=0
x = 0
l = 0
print("started ....")
while True:
    c = f.read(1)
    if not c:
        break
    if c == '\n':
        i = i + 1
        j = 0
    elif c!=' ':
        if c =='.':
            l=0
            continue
        x = x*10 + (ord(c) - 48)
        l += 1
    elif c == ' ':
        cent[i][j] =  x * math.pow(10,-l)
        j = j + 1
        x = 0
print("copying completed .....")
f.close()

import math as m
def dist(A,B):
    x = np.zeros([32])
    for i in range(32):
        x[i] = A[i]-B[i]
   
    return m.sqrt(sum(x*x))
cap = cv2.VideoCapture(path)
ret = True
k = -1
while ret:
	ret,img = cap.read() 
	k = k + 1	
	if ret == False:
		cap.release()
		break
	#num_rows, num_cols = img.shape[:2]
	#rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2),30, 1)
	#query= cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
	gray_query = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	orb = cv2.ORB_create()
	kp_1 = orb.detect(gray_query, None)
	kp_1, feat = orb.compute(gray_query, kp_1)
	hist2 = np.zeros([30])
	for i in range(500):
		mat_a = []
		mat_b = []
		a = dist(cent[0],feat[i])
		b = dist(cent[1],feat[i])
		if(a < b):
		    mat_a.append(feat[i])
		else:
		    mat_b.append(feat[i])
		   
		mat_a = np.array(mat_a)
		mat_b = np.array(mat_b)
		mat_a_1 = []
		mat_a_2 = []
		mat_a_3 = []
		mat_a_4 = []
		mat_a_5 = []
		for j in range(len(mat_a)):
		    a = dist(cent[2],mat_a[j])
		    b = dist(cent[3],mat_a[j])
		    c = dist(cent[4],mat_a[j])
		    d = dist(cent[5],mat_a[j])
		    e = dist(cent[6],mat_a[j])
		   
		    if (a<b) and (a<c) and (a<d) and (a<e):
		        mat_a_1.append(mat_a[j])
		    elif (b<c) and (b<d) and (b<e):
		        mat_a_2.append(mat_a[j])
		    elif (c<d) and (c<e):
		        mat_a_3.append(mat_a[j])
		    elif (d<e):
		        mat_a_4.append(mat_a[j])
		    else:
		        mat_a_5.append(mat_a[j])
	   
		mat_a_1 = np.array(mat_a_1)
		mat_a_2 = np.array(mat_a_2)
		mat_a_3 = np.array(mat_a_3)
		mat_a_4 = np.array(mat_a_4)
		mat_a_5 = np.array(mat_a_5)
	   
		for j in range(len(mat_a_1)):
		    a = dist(cent[7],mat_a_1[j])
		    b = dist(cent[8],mat_a_1[j])
		    c = dist(cent[9],mat_a_1[j])
		   
		    if (a<b) and (a<c):
		        hist2[0] += 1
		    elif (b<c):
		        hist2[1] += 1
		    else:
		        hist2[2] += 1
		       
		for j in range(len(mat_a_2)):
		    a = dist(cent[10],mat_a_2[j])
		    b = dist(cent[11],mat_a_2[j])
		    c = dist(cent[12],mat_a_2[j])
		   
		    if (a<b) and (a<c):
		        hist2[3] += 1
		    elif (b<c):
		        hist2[4] += 1
		    else:
		        hist2[5] += 1
		       
		for j in range(len(mat_a_3)):
		    a = dist(cent[13],mat_a_3[j])
		    b = dist(cent[14],mat_a_3[j])
		    c = dist(cent[15],mat_a_3[j])
		   
		    if (a<b) and (a<c):
		        hist2[6] += 1
		    elif (b<c):
		        hist2[7] += 1
		    else:
		        hist2[8] += 1
		       
		for j in range(len(mat_a_4)):
		    a = dist(cent[16],mat_a_4[j])
		    b = dist(cent[17],mat_a_4[j])
		    c = dist(cent[18],mat_a_4[j])
		   
		    if (a<b) and (a<c):
		        hist2[9] += 1
		    elif (b<c):
		        hist2[10] += 1
		    else:
		        hist2[11] += 1
		       
		for j in range(len(mat_a_5)):
		    a = dist(cent[19],mat_a_5[j])
		    b = dist(cent[20],mat_a_5[j])
		    c = dist(cent[21],mat_a_5[j])
		   
		    if (a<b) and (a<c):
		        hist2[12] += 1
		    elif (b<c):
		        hist2[13] += 1
		    else:
		        hist2[14] += 1
		       
		mat_b_1 = []
		mat_b_2 = []
		mat_b_3 = []
		mat_b_4 = []
		mat_b_5 = []
		for j in range(len(mat_b)):
		    a = dist(cent[22],mat_b[j])
		    b = dist(cent[23],mat_b[j])
		    c = dist(cent[24],mat_b[j])
		    d = dist(cent[25],mat_b[j])
		    e = dist(cent[26],mat_b[j])
		   
		    if (a<b) and (a<c) and (a<d) and (a<e):
		        mat_b_1.append(mat_b[j])
		    elif (b<c) and (b<d) and (b<e):
		        mat_b_2.append(mat_b[j])
		    elif (c<d) and (c<e):
		        mat_b_3.append(mat_b[j])
		    elif (d<e):
		        mat_b_4.append(mat_b[j])
		    else:
		        mat_b_5.append(mat_b[j])
	   
		mat_b_1 = np.array(mat_b_1)
		mat_b_2 = np.array(mat_b_2)
		mat_b_3 = np.array(mat_b_3)
		mat_b_4 = np.array(mat_b_4)
		mat_b_5 = np.array(mat_b_5)
	   
		for j in range(len(mat_b_1)):
		    a = dist(cent[27],mat_b_1[j])
		    b = dist(cent[28],mat_b_1[j])
		    c = dist(cent[29],mat_b_1[j])
		   
		    if (a<b) and (a<c):
		        hist2[15] += 1
		    elif (b<c):
		        hist2[16] += 1
		    else:
		        hist2[17] += 1
		       
		for j in range(len(mat_b_2)):
		    a = dist(cent[30],mat_b_2[j])
		    b = dist(cent[31],mat_b_2[j])
		    c = dist(cent[32],mat_b_2[j])
		   
		    if (a<b) and (a<c):
		        hist2[18] += 1
		    elif (b<c):
		        hist2[19] += 1
		    else:
		        hist2[20] += 1
		       
		for j in range(len(mat_b_3)):
		    a = dist(cent[33],mat_b_3[j])
		    b = dist(cent[34],mat_b_3[j])
		    c = dist(cent[35],mat_b_3[j])
		   
		    if (a<b) and (a<c):
		        hist2[21] += 1
		    elif (b<c):
		        hist2[22] += 1
		    else:
		        hist2[23] += 1
		       
		for j in range(len(mat_b_4)):
		    a = dist(cent[36],mat_b_4[j])
		    b = dist(cent[37],mat_b_4[j])
		    c = dist(cent[38],mat_b_4[j])
		   
		    if (a<b) and (a<c):
		        hist2[24] += 1
		    elif (b<c):
		        hist2[25] += 1
		    else:
		        hist2[26] += 1
		       
		for j in range(len(mat_b_5)):
		    a = dist(cent[39],mat_b_5[j])
		    b = dist(cent[40],mat_b_5[j])
		    c = dist(cent[41],mat_b_5[j])
		   
		    if (a<b) and (a<c):
		        hist2[27] += 1
		    elif (b<c):
		        hist2[28] += 1
		    else:
		        hist2[29] += 1
	print("Checking the accuracy .... ")
	diff = np.zeros([n])
	for i in range(n):
		l = 0
		for j in range(30):
		    l = l + abs(hist2[j]-hist[i][j])
		diff[i]=l

	for i in range(n):
		if diff[i]==min(diff):
		    print(i,k)
		    break

end =t.time()
print("Time Taken: ", str(((end - start))))

