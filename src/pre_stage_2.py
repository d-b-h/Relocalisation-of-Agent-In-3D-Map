import cv2
import sys
path = sys.argv[1]
cap = cv2.VideoCapture(path)
l=0
print("started.........")
ret = True
while ret:
    ret,frame = cap.read()
    if ret:
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
        cv2.imwrite("/home/dikshit/Desktop/frames/frame_%d.jpg"%l,frame)
        l += 1
        print("frame %d"%l)
    else:
        cap.release()
print("completed.......")
cv2.destroyAllWindows()

