# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 20:03:33 2018

@author: hp
"""

import cv2
import numpy as np
def draw_matches(vis, keypoints0, keypoints1):
    for kp in range(keypoints0.shape[0]):
            xy0 = keypoints0[kp,:]
            xy1 = keypoints1[kp,:]
            cv2.circle(vis, (int(xy0[0]), int(xy0[1])), 2, (255,0,0))
            cv2.circle(vis, (int(xy1[0]), int(xy1[1])), 2, (0,255,0))
            cv2.line(vis, (int(xy0[0]), int(xy0[1])), (int(xy1[0]), int(xy1[1])), (0,0,255), 1)


def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in range(keypoints.shape[0]):
            xy = keypoints[kp,:]
            cv2.circle(vis, (int(xy[0]), int(xy[1])), 2, color)

def get_P_prime_from_F(F):
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Z = np.array([[0,-1,0],[1,0,0],[0,0,0]])
    U, D, V = np.linalg.svd(F)
    S = np.dot(np.dot(U,Z),U.T)
    M = np.dot(np.dot(np.dot(U,W.T),np.diag(D)),V)
    # import pdb; pdb.set_trace()
    assert(np.allclose(F,np.dot(np.dot(U,np.diag(D)),V)))
    assert(np.allclose(F,np.dot(S,M)))
    P_prime = np.hstack((M, U[:,-1].reshape((3,1))))
    return P_prime


FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10

SGBM=0
maxDiff=32
blockSize=21
stereoMatcher=cv2.StereoBM_create(maxDiff,blockSize)
detector=cv2.ORB_create( nfeatures = 4000 )
matcher=cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)

img1=cv2.imread('frame_7.jpg')
img2=cv2.imread('frame_8.jpg')
w,h=img1.shape[:2]
prev_keypoints, prev_descrs=detector.detectAndCompute(img1.copy(), None)
curr_keypoints, curr_descrs =detector.detectAndCompute(img2.copy(), None)
matcher.clear()
matcher.add([prev_descrs.astype(np.uint8)])
matches=matcher.knnMatch(curr_descrs, k = 2)

matches=[m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
print('%d matches.' % len(matches))
p0=[prev_keypoints[m.trainIdx].pt for m in matches]
p1=[curr_keypoints[m.queryIdx].pt for m in matches]
p0,p1=np.float32((p0, p1))

#Estimate homography
H,status=cv2.findHomography(p0, p1, cv2.RANSAC, 13.0)
status = status.ravel() != 0
print('inliner percentage: %.1f %%' % (status.mean().item(0)*100.,))
#if(status.sum()<MIN_MATCH_COUNT):
#   continue
p0, p1 = p0[status], p1[status]

# Display inliners
imgpair = cv2.addWeighted(img2, .5, img1, .5, 0)
draw_matches(imgpair, p0, p1)
cv2.imshow('keypoint matches', imgpair)

# Estimate fundamental matrix
F, status = cv2.findFundamentalMat(p0, p1, cv2.FM_8POINT, 3, .99)
# Estimate camera matrix
p0 = np.hstack((p0, np.ones((p0.shape[0],1)))).astype(np.float32)
K = cv2.initCameraMatrix2D([p0], [p1], (w, h))
p0 = p0[:,:2]
print(K)            
# Estimate essential matrix
E, status = cv2.findEssentialMat(p0, p1, cameraMatrix=K)
ret, R, t, status = cv2.recoverPose(E, p0, p1, cameraMatrix=K, mask = status)
rvec, jacobian = cv2.Rodrigues(R)
print('(R, t)=', rvec.T, '\n', t.T)


