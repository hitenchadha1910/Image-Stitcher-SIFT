import cv2
import numpy as np

imgA = cv2.imread('image_right.jpg')
#imgA = cv2.imread('original_image_left.jpg')
# imgA = cv2.resize(imgA, (0,0), fx=0.3, fy=0.3)
img1 = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)

imgB = cv2.imread('image_left.jpg')
#imgB = cv2.imread('original_image_right.jpg')
# imgB = cv2.resize(imgB, (0,0), fx=0.3, fy=0.3)
img2 = cv2.cvtColor(imgB,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# find key points
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(imgA,kp1,None))

match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2)

good = []
#for computing best matches for image stitching purpose
for m,n in matches:
    if m.distance < 0.03*n.distance:
        good.append(m)

#draw matching parameters
draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       flags=2)

img3 = cv2.drawMatches(imgA,kp1,imgB,kp2,good,None,**draw_params)
# cv2.imshow("original_image_drawMatches.jpg", img3)
# cv2.waitKey(8000)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    #converting interest pts list to use as arguments for homography function
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    #cv2.imshow("original_image_overlapping.jpg", img2)
else:
    print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))

#Warp right image on left image with homography matrix
dst = cv2.warpPerspective(imgA,M,(imgB.shape[1] + imgA.shape[1], imgB.shape[0]))
dst[0:imgB.shape[0],0:imgB.shape[1]] = imgB
# cv2.imshow("original_image_stitched.jpg", dst)

#trimming the unwanted black part in panorama
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

# cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
# cv2.waitKey(8000)
cv2.imwrite("image_stitched_crop.jpg", trim(dst))