import numpy as np
from imutils import face_utils
import cv2
import dlib
from scipy.spatial import distance as dist
import time
from firebase import firebase


FBconn = firebase.FirebaseApplication('https://ed-workshop.firebaseio.com/', None)


def MAR(mouth):
    return (dist.euclidean(mouth[0],mouth[1]) + dist.euclidean(mouth[2],mouth[3]) + dist.euclidean(mouth[4], mouth[5]))/(3*dist.euclidean(mouth[6], mouth[7]))

def N_MID(nose):
    return nose.mean(axis=0)

detect = dlib.get_frontal_face_detector()
file = "/media/khurshed2504/Data/ED Workshop/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(file)
(mst, mend) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(ns, nend) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]

total_devices = 4

for i in range(1,total_devices+1):
    FBconn.put('/state', '/', int(10*i))

rad =  70
nose_pts_x = []
nose_pts_y = []
mars = []
nose_pose_x = 0
nose_pose_y = 0

state_space = ['OFF', 'ON']

start = time.time()


cap = cv2.VideoCapture(0)
while time.time() - start < 10:
    _, image = cap.read()
    image = cv2.flip(image,1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detect(gray, 0)
    for (i,rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        nose = shape[31:36]
        nose_pt = N_MID(nose)
        nose_pts_x.append(nose_pt[0])
        nose_pts_y.append(nose_pt[1])
        nose_pose_x = np.mean(nose_pts_x)
        nose_pose_y = np.mean(nose_pts_y)
        nose_roi = shape[ns: nend]
        nose_hull = cv2.convexHull(nose_roi)
        cv2.drawContours(image, [nose_hull], -1, (0, 255, 0), 1)

        m_ind = [50, 58, 51, 57, 52, 56, 48, 54]
        mouth = shape[m_ind]
        mouth_roi = shape[mst: mend]
        mouth_hull = cv2.convexHull(mouth_roi)
        cv2.drawContours(image, [mouth_hull], -1, (0, 255, 0), 1)
        mars.append(MAR(mouth))
        mar_mean = np.mean(mars)
    board = np.zeros((200, 640, 3), dtype=np.uint8)
    cv2.putText(board, "Open your mouth and keep your nose stable", (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
    cv2.putText(board, "Calibration ON", (225,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
    res = np.vstack((image, board))
    cv2.imshow('Calibration', res)
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


print("Mean Nose Position: ", nose_pose_x, nose_pose_y)
print("Mean Mouth Aspect Ratio: ", mar_mean)

cap = cv2.VideoCapture(0)
rcnt = 0
lcnt = 0
dev_no = 1
ptr = 1
nose_pts_x = []
nose_pts_y = []
dev_arr = np.arange(1,total_devices+1)
dev_states = np.zeros(total_devices)
ut = 0.8
fcnt = 0
min_device_change_frames = 12
min_toggle_frames = 15

while True:
    _, image = cap.read()
    image = cv2.flip(image,1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detect(gray, 0)

    for (i,rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        m_ind = [50, 58, 51, 57, 52, 56, 48, 54]
        mouth = shape[m_ind]
        nose = shape[31:36]
        nose_pt = N_MID(nose)
        nose_x = nose_pt[0]
        if nose_x > nose_pose_x + rad:
            rcnt += 1
        if nose_x < nose_pose_x - rad:
            lcnt += 1
        if rcnt > min_device_change_frames:
            rcnt = 0
            ptr += 1
            dev_no = dev_arr[(ptr%total_devices) - 1]
            print("Selected Device: ",dev_no)
            print("Current State: ", state_space[int(dev_states[dev_no-1])])
        if lcnt > min_device_change_frames:
            lcnt = 0
            ptr -= 1
            dev_no = dev_arr[(ptr%total_devices) - 1]
            print("Selected Device: ", dev_no)
            print("Current State: ", state_space[int(dev_states[dev_no-1])])

        
        mouth_roi = shape[mst: mend]
        nose_roi = shape[ns: nend]
        mouth_hull = cv2.convexHull(mouth_roi)
        nose_hull = cv2.convexHull(nose_roi)
        cv2.drawContours(image, [mouth_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [nose_hull], -1, (0, 255, 0), 1)
        mar = MAR(mouth)
        if mar > mar_mean*ut:
            fcnt += 1
        if fcnt > min_toggle_frames:
            fcnt = 0
            dev_states[dev_no-1] = 1 - dev_states[dev_no-1]
            print("Device Number: {}, State: {}".format(dev_no, state_space[int(dev_states[dev_no-1])]))
            data = int(10*dev_no + dev_states[dev_no-1])
            FBconn.put('/state', '/', data)
            
    cv2.circle(image, (int(nose_pose_x), int(nose_pose_y)), rad, (255,0,0), 1)    
    cv2.imshow('Image', image)
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        cap.release()
        break
cv2.destroyAllWindows()
cap.release()