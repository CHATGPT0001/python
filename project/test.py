import argparse
import time
from collections import OrderedDict
import cv2
import dlib
import imutils
import numpy as np
from scipy.spatial import distance as dist
import threading
import json
import requests
import base64
import os

# 全局变量来存储时间
URL=f"http://172.20.10.3/DrowsyDrivingService/DDService.ashx?Action="
elapsed_time_str = ""
hour = 0
minute = 0
second = 0
tempEtotal = 0
tempMtotal = 0
requestData = {
    "MethodName": "SetDetectData",
    "TotalBlinkTimes": 0,
    "BlinkTimesPerMinutes": 0,
    "TotalYawnTimes": 0,
    "YawnTimesPerMinutes": 0#新增
}

def toBase64(response):
    json_response = json.dumps(response, indent=4)
    encoded_string = base64.b64encode(json_response.encode())
    return encoded_string.decode()

def checkWeb(base64_encoded_string):
    global URL
    try:
        url = f"{URL}{base64_encoded_string}"#改IP
        response = requests.get(url)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to make request: {e}")
        return None

def original(base64_string):
    decoded_bytes = base64.b64decode(base64_string)
    json_string = decoded_bytes.decode()
    response = json.loads(json_string)
    return response

def time_start():
    global elapsed_time_str, hour, minute, second
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        elapsed_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        hour = hours
        minute = minutes
        second = seconds
        time.sleep(1)

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)

def shape_to_np(shape):
    coords = np.zeros((shape.num_parts, 2), dtype="int")
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def process_video_stream(vs, detector, predictor, lStart, lEnd, rStart, rEnd, mStart, mEnd):
    global ECOUNTER, ETOTAL, MCOUNTER, MTOTAL, elapsed_time_str, tempEtotal, second,tempMtotal

    while True:
        frame = vs.read()[1]
        if frame is None:
            break
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
        
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
        
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(mouth)
        
            if ear < EYE_EAR_THRESH :
                ECOUNTER += 1
            else:
                if ECOUNTER >= EYE_CONSEC_FRAMES:
                    ETOTAL += 1
                    tempEtotal += 1
                ECOUNTER = 0   
            if (second % 60) == 0:
                tempEtotal = 0        
        
            if mar > MOUTH_MAR_THRESH:
                MCOUNTER += 1
            else:
                if MCOUNTER >= MOUTH_CONSEC_FRAMES:
                    MTOTAL += 1
                    tempMtotal+= 1
                MCOUNTER = 0
            if (second % 60) == 0:
                tempMtotal = 0       
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)
        
            cv2.putText(frame, f"Blinks: {ETOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"PerMinuteBlinks: {tempEtotal}", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Yawn: {MTOTAL}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"PerMinuteYawns: {tempMtotal}", (600, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, elapsed_time_str, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            requestData = {
                "MethodName": "SetDetectData",
                "TotalBlinkTimes": ETOTAL,
                "BlinkTimesPerMinutes": tempEtotal,
                "TotalYawnTimes": MTOTAL,
                "YawnTimesPerMinutes": tempMtotal#新增
            }
            if second % 5 == 0:  # 每五秒传输一次
                encoded_response = toBase64(requestData)
                response = checkWeb(encoded_response)
                decoded_response = original(response.text)
                print(decoded_response)
                
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    vs.release()
    cv2.destroyAllWindows()

def check_detect_flag():
    global URL
    requestData = {
        "MethodName": "GetDetectFlag"
    }
    request = toBase64(requestData)
    url = URL+f"{request}"#改IP
    
    while True:
        try:
            response = requests.get(url)
            response.raise_for_status()
            print(response.text)
            
            decoded_response = original(response.text)
            print(decoded_response)
            status= decoded_response.get("Status")
            detectflag=decoded_response.get("Message")
            
            if status == "Success":
                if detectflag.lower() =="true":                
                    print("[INFO] DetectFlag is true, starting the recognition system...")
                    break
                else:
                    print("[INFO] DetectFlag is false, waiting...")
                    time.sleep(5)# 等待5秒后再检查一次
            else:
                print("[INFO] Server Error, waiting...")
                time.sleep(5)  # 等待5秒后再检查一次
                
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to get DetectFlag: {e}")
            time.sleep(5)  # 如果请求失败，等待5秒后再尝试
            continue
# 检查 DetectFlag 是否为 true
check_detect_flag()

# 設置閾值
EYE_EAR_THRESH = 0.22
EYE_CONSEC_FRAMES = 1.5
MOUTH_MAR_THRESH = 0.74
MOUTH_CONSEC_FRAMES = 1.5

# 初始化计数器
ECOUNTER, ETOTAL = 0, 0
MCOUNTER, MTOTAL = 0, 0

# 加载检测器和预测器
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
shape_predictor_path = os.path.join(
    "D:/eclipse/eclipse/project", "shape_predictor_68_face_landmarks.dat"
)

predictor = dlib.shape_predictor(shape_predictor_path)

FACIAL_LANDMARK_68_IDXS = OrderedDict([
    ('mouth', (48, 68)), ('right_eye', (36, 42)), ('left_eye', (42, 48))
])

(lStart, lEnd) = FACIAL_LANDMARK_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARK_68_IDXS["right_eye"]
(mStart, mEnd) = FACIAL_LANDMARK_68_IDXS["mouth"]

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(1)  # 0是电脑镜头
time.sleep(1.0)

# 创建和启动计时器线程
timer_thread = threading.Thread(target=time_start)
timer_thread.daemon = True  # 设置为守护线程，这样主程序退出时线程也会自动退出
timer_thread.start()

# 处理视频流
process_video_stream(vs, detector, predictor, lStart, lEnd, rStart, rEnd, mStart, mEnd)
