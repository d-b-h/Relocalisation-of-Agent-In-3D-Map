# Relocalisation-of-Agent-In-3D-Map

Deepti Hegde, Ramesh Tabib, Uma Mudenagudi 7th National Conference on Computer Vision,
Pattern Recognition, Image Processing and Graphics (NCVPRIPG 2019)

Details may be seen in presReloc.pdf.

Dataset used: EuRoC Machine Lab 1, TUM frxyz1 and frxyz2

Implement relocalisation of agent (camera) in real time.
b. Produce camera pose and trajectory given query image of location 
after 3D map has been generated using SLAM.
c. Project is of three stages:
	1. 3D map generation using SLAM
	2. Creation of unique dictionary of visual words for given dataset.
	3. Large scale image search using vocabulary tree.
 
 
Implementation
1. Stage 1 :- SLAM
2. Stage 2 :- Visual Word Dictionary
3. Stage 3 :- Image Search
Details of Implementation

Main objective is to create a relocalisation system which is as fast
and efficient as possible, and able to be implemented in real time.


Steps to Implement
 
1. Calibrate camera and save parameters.
2. Capture a video of area to be mapped.
3. Run ORB slam
4. Run stage_2.py
5. In real time, run stage_3.py to relocalise and obtain camera position in space, 
and map if required.


Pre-requisites to run our code (Python)

1. Opencv 3.4.1
2. Pangollin
3. Eigen
4. OpenGL
5. Matlab (if plotting is required)
