import cv2
import numpy as np
import os
import shutil
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time as t
import sys
script = sys.argv[0]
path = sys.argv[1]
if os.path.exists(path):
	start = t.time()
	cap = cv2.VideoCapture(path)	
	os.mkdir("frames")
	l=0
	print("started.........")
	ret = True
	while ret:
		ret,frame = cap.read()
		if ret:
		    cv2.imwrite("frames/frame_%d.jpg"%l,frame)
		    l += 1
		else:
		    cap.release()
	n = l
	print("Creating the Feature matrix")
	mat = np.zeros([n*500,32])
	print(n)
	l = 0
	for l in range(n):
		input_image = cv2.imread("frames/frame_%d.jpg"%l)
		gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
		orb = cv2.ORB_create()
		kp = orb.detect(gray_image, None)
		kp, dp = orb.compute(gray_image, kp)
		for i in range(len(dp)):
			for j in range(32):
				mat[i+(500*l)][j]=dp[i][j]

	print("done with matrix")

	print("Creating the centroids for the Hierarchical clustering")
	kmeans_root = KMeans(n_clusters=2,max_iter = 500)
	kmeans_a = KMeans(n_clusters=5,max_iter = 500)
	kmeans_b = KMeans(n_clusters=5,max_iter = 500)
	kmeans_a_1 = KMeans(n_clusters=3,max_iter = 500)
	kmeans_a_2 = KMeans(n_clusters=3,max_iter = 500)
	kmeans_a_3 = KMeans(n_clusters=3,max_iter = 500)
	kmeans_a_4 = KMeans(n_clusters=3,max_iter = 500)
	kmeans_a_5 = KMeans(n_clusters=3,max_iter = 500)
	kmeans_b_1 = KMeans(n_clusters=3,max_iter = 500)
	kmeans_b_2 = KMeans(n_clusters=3,max_iter = 500)
	kmeans_b_3 = KMeans(n_clusters=3,max_iter = 500)
	kmeans_b_4 = KMeans(n_clusters=3,max_iter = 500)
	kmeans_b_5 = KMeans(n_clusters=3,max_iter = 500)

	kmeans_root.fit(mat)
	labels_r = kmeans_root.predict(mat)
	centroids_r = kmeans_root.cluster_centers_

	mat_a=[]
	mat_b=[]
	for i in range(n*500):
		if labels_r[i]==0:
		    mat_a.append(mat[i])
		else:
		    mat_b.append(mat[i])

	del labels_r
	mat_a = np.array(mat_a)
	mat_b = np.array(mat_b)
	kmeans_a.fit(mat_a)
	labels_a = kmeans_a.predict(mat_a)
	centroids_a= kmeans_a.cluster_centers_
	mat_a_1 = []
	mat_a_2 = []
	mat_a_3 = []
	mat_a_4 = []
	mat_a_5 = []
	for i in range(len(mat_a)):
		if labels_a[i]==0:
		    mat_a_1.append(mat_a[i])
		elif labels_a[i] == 1:
		    mat_a_2.append(mat_a[i])
		elif labels_a[i] == 2:
		    mat_a_3.append(mat_a[i])
		elif labels_a[i] == 3:
		    mat_a_4.append(mat_a[i])
		else:
		    mat_a_5.append(mat_a[i])
	del mat_a,labels_a
	mat_a_1=np.array(mat_a_1)
	mat_a_2=np.array(mat_a_2)
	mat_a_3=np.array(mat_a_3)
	mat_a_4=np.array(mat_a_4)
	mat_a_5=np.array(mat_a_5)

	kmeans_b.fit(mat_b)
	labels_b = kmeans_b.predict(mat_b)
	centroids_b = kmeans_b.cluster_centers_
	mat_b_1 = []
	mat_b_2 = []
	mat_b_3 = []
	mat_b_4 = []
	mat_b_5 = []
	for i in range(len(mat_b)):
		if labels_b[i]==0:
		    mat_b_1.append(mat_b[i])
		elif labels_b[i] == 1:
		    mat_b_2.append(mat_b[i])
		elif labels_b[i] == 2:
		    mat_b_3.append(mat_b[i])
		elif labels_b[i] == 3:
		    mat_b_4.append(mat_b[i])
		else:
		    mat_b_5.append(mat_b[i])
		    
	del mat_b,labels_b

	mat_b_1=np.array(mat_b_1)
	mat_b_2=np.array(mat_b_2)
	mat_b_3=np.array(mat_b_3)
	mat_b_4=np.array(mat_b_4)
	mat_b_5=np.array(mat_b_5)

	kmeans_a_1.fit(mat_a_1)
	labels_a_1 = kmeans_a_1.predict(mat_a_1)
	centroids_a_1= kmeans_a_1.cluster_centers_

	kmeans_a_2.fit(mat_a_2)
	labels_a_2 = kmeans_a_2.predict(mat_a_2)
	centroids_a_2= kmeans_a_2.cluster_centers_

	kmeans_a_3.fit(mat_a_3)
	labels_a_3 = kmeans_a_3.predict(mat_a_3)
	centroids_a_3= kmeans_a_3.cluster_centers_

	kmeans_a_4.fit(mat_a_4)
	labels_a_4 = kmeans_a_4.predict(mat_a_4)
	centroids_a_4= kmeans_a_4.cluster_centers_

	kmeans_a_5.fit(mat_a_5)
	labels_a_5 = kmeans_a_5.predict(mat_a_5)
	centroids_a_5= kmeans_a_5.cluster_centers_

	kmeans_b_1.fit(mat_b_1)
	labels_b_1 = kmeans_b_1.predict(mat_b_1)
	centroids_b_1= kmeans_b_1.cluster_centers_

	kmeans_b_2.fit(mat_b_2)
	labels_b_2 = kmeans_b_2.predict(mat_b_2)
	centroids_b_2= kmeans_b_2.cluster_centers_

	kmeans_b_3.fit(mat_b_3)
	labels_b_3 = kmeans_b_3.predict(mat_b_3)
	centroids_b_3= kmeans_b_3.cluster_centers_

	kmeans_b_4.fit(mat_b_4)
	labels_b_4 = kmeans_b_4.predict(mat_b_4)
	centroids_b_4= kmeans_b_4.cluster_centers_

	kmeans_b_5.fit(mat_b_5)
	labels_b_5 = kmeans_b_5.predict(mat_b_5)
	centroids_b_5= kmeans_b_5.cluster_centers_

	del labels_a_1,labels_a_2,labels_a_3,labels_a_4,labels_a_5,labels_b_1,labels_b_2,labels_b_3,labels_b_4,labels_b_5
	print("Done with clustering")
	print("Creating the Histogram of each fraames")
	hist = np.zeros([n,30],int)
	feat = np.zeros([500,32])
	for i in range(n):
		mat_b_1 = []
		mat_b_2 = []
		mat_b_3 = []
		mat_b_4 = []
		mat_b_5 = []
		mat_a_1 = []
		mat_a_2 = []
		mat_a_3 = []
		mat_a_4 = []
		mat_a_5 = []
		mat_a=[]
		mat_b=[]
		feat = mat[i*500 : (i*500)+500]
		labels_r = kmeans_root.predict(feat)
		for j in range(500):
			if labels_r[j] == 0:
				mat_a.append(feat[j])
			else:
				mat_b.append(feat[j])

		mat_a = np.array(mat_a)
		mat_b = np.array(mat_b)
		labels_a = kmeans_a.predict(mat_a)
		for j in range(len(mat_a)):
			if labels_a[j]==0:
				mat_a_1.append(mat_a[j])
			elif labels_a[j] == 1:
				mat_a_2.append(mat_a[j])
			elif labels_a[j] == 2:
				mat_a_3.append(mat_a[j])
			elif labels_a[j] == 3:
				mat_a_4.append(mat_a[j])
			else:
				mat_a_5.append(mat_a[j])

		mat_a_1=np.array(mat_a_1)
		mat_a_2=np.array(mat_a_2)
		mat_a_3=np.array(mat_a_3)
		mat_a_4=np.array(mat_a_4)
		mat_a_5=np.array(mat_a_5) 

		labels_b = kmeans_b.predict(mat_b)
		for j in range(len(mat_b)):
			if labels_b[j]==0:
				mat_b_1.append(mat_b[j])
			elif labels_b[j] == 1:
				mat_b_2.append(mat_b[j])
			elif labels_b[j] == 2:
				mat_b_3.append(mat_b[j])
			elif labels_b[j] == 3:
				mat_b_4.append(mat_b[j])
			else:
				mat_b_5.append(mat_b[j])

		mat_b_1=np.array(mat_b_1)
		mat_b_2=np.array(mat_b_2)
		mat_b_3=np.array(mat_b_3)
		mat_b_4=np.array(mat_b_4)
		mat_b_5=np.array(mat_b_5)

		labels_a_1 = kmeans_a_1.predict(mat_a_1)
		labels_a_2 = kmeans_a_2.predict(mat_a_2)
		labels_a_3 = kmeans_a_3.predict(mat_a_3)
		labels_a_4 = kmeans_a_4.predict(mat_a_4)
		labels_a_5 = kmeans_a_5.predict(mat_a_5)
		labels_b_1 = kmeans_b_1.predict(mat_b_1)
		labels_b_2 = kmeans_b_2.predict(mat_b_2)
		labels_b_3 = kmeans_b_3.predict(mat_b_3)
		labels_b_4 = kmeans_b_4.predict(mat_b_4)
		labels_b_5 = kmeans_b_5.predict(mat_b_5)

		for j in range(len(labels_a_1)):
			if labels_a_1[j] == 0:
				hist[i][0] += 1
			elif labels_a_1[j] == 1:
				hist[i][1] += 1
			else:
				hist[i][2] += 1
		
		for j in range(len(labels_a_2)):
			if labels_a_2[j] == 0:
				hist[i][3] += 1
			elif labels_a_2[j] == 1:
				hist[i][4] += 1
			else:
				hist[i][5] += 1
		
		for j in range(len(labels_a_3)):
			if labels_a_3[j] == 0:
				hist[i][6] += 1
			elif labels_a_3[j] == 1:
				hist[i][7] += 1
			else:
				hist[i][8] += 1
		
		for j in range(len(labels_a_4)):
			if labels_a_4[j] == 0:
				hist[i][9] += 1
			elif labels_a_4[j] == 1:
				hist[i][10] += 1
			else:
				hist[i][11] += 1
		
		for j in range(len(labels_a_5)):
			if labels_a_5[j] == 0:
				hist[i][12] += 1
			elif labels_a_5[j] == 1:
				hist[i][13] += 1
			else:
				hist[i][14] += 1
		
		for j in range(len(labels_b_1)):
			if labels_b_1[j] == 0:
				hist[i][15] += 1
			elif labels_b_1[j] == 1:
				hist[i][16] += 1
			else:
				hist[i][17] += 1
		
		for j in range(len(labels_b_2)):
			if labels_b_2[j] == 0:
				hist[i][18] += 1
			elif labels_b_2[j] == 1:
				hist[i][19] += 1
			else:
				hist[i][20] += 1
		
		for j in range(len(labels_b_3)):
			if labels_b_3[j] == 0:
				hist[i][21] += 1
			elif labels_b_3[j] == 1:
				hist[i][22] += 1
			else:
				hist[i][23] += 1
		
		for j in range(len(labels_b_4)):
			if labels_b_4[j] == 0:
				hist[i][24] += 1
			elif labels_b_4[j] == 1:
				hist[i][25] += 1
			else:
				hist[i][26] += 1
		
		for j in range(len(labels_b_5)):
			if labels_b_5[j] == 0:
				hist[i][27] += 1
			elif labels_b_5[j] == 1:
				hist[i][28] += 1
			else:
				hist[i][29] += 1
		
	del labels_a_1,labels_a_2,labels_a_3,labels_a_4,labels_a_5,labels_b_1,labels_b_2,labels_b_3,labels_b_4,labels_b_5
	del labels_r,labels_a,labels_b
	del mat_a_1,mat_a_2,mat_a_3,mat_a_4,mat_a_5,mat_b_1,mat_b_2,mat_b_3,mat_b_4,mat_b_5,mat_a,mat_b
	print("Done with histogram of each frame")
	print("Saving the histogram values and centroids in a file")
	f = open("centroid.txt","w")
	for i in range(2):
		for j in range(32):
		    f.write(str((centroids_r[i][j])))
		    f.write(" ")
		f.write("\n")
	for i in range(5):
		for j in range(32):
		    f.write(str((centroids_a[i][j])))
		    f.write(" ")
		f.write("\n")
	for i in range(3):
		for j in range(32):
		    f.write(str((centroids_a_1[i][j])))
		    f.write(" ")
		f.write("\n")
	for i in range(3):
		for j in range(32):
		    f.write(str((centroids_a_2[i][j])))
		    f.write(" ")
		f.write("\n")
	for i in range(3):
		for j in range(32):
		    f.write(str((centroids_a_3[i][j])))
		    f.write(" ")
		f.write("\n")
	for i in range(3):
		for j in range(32):
		    f.write(str((centroids_a_4[i][j])))
		    f.write(" ")
		f.write("\n")
	for i in range(3):
		for j in range(32):
		    f.write(str((centroids_a_5[i][j])))
		    f.write(" ")
		f.write("\n")

	for i in range(5):
		for j in range(32):
		    f.write(str((centroids_b[i][j])))
		    f.write(" ")
		f.write("\n")
	for i in range(3):
		for j in range(32):
		    f.write(str((centroids_b_1[i][j])))
		    f.write(" ")
		f.write("\n")
	for i in range(3):
		for j in range(32):
		    f.write(str((centroids_b_2[i][j])))
		    f.write(" ")
		f.write("\n")
	for i in range(3):
		for j in range(32):
		    f.write(str((centroids_b_3[i][j])))
		    f.write(" ")
		f.write("\n")
	for i in range(3):
		for j in range(32):
		    f.write(str((centroids_b_4[i][j])))
		    f.write(" ")
		f.write("\n")
	for i in range(3):
		for j in range(32):
		    f.write(str((centroids_b_5[i][j])))
		    f.write(" ")
		f.write("\n")

	f.close()

	histFile = open("corpusHist.txt","w")
	histFile.write(str(n))
	histFile.write("\n")
	for i in range(n):
		for j in range(30):
		    histFile.write(str((hist[i][j])))
		    histFile.write(" ")
		histFile.write("\n")

	histFile.close()
	end = t.time()
	shutil.rmtree("frames")
	print("Done with the saving the required points and data")
	print("Time Taken: ", str(((end - start))))
	print("Done with Stage_1")
else:
	print("wrong path")
