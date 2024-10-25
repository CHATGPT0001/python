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
import sys  # 用於安全退出程序

# 全局變數
URL = "http://172.20.10.3/DrowsyDrivingService/DDService.ashx?Action="
elapsed_time_str = ""
hour = 0
minute = 0
second = 0
tempEtotal = 0
tempMtotal = 0
vs = None  # 全局攝像頭變數

requestData = {
    "MethodName": "SetDetectData",
    "TotalBlinkTimes": 0,
    "BlinkTimesPerMinutes": 0,
    "TotalYawnTimes": 0,
    "YawnTimesPerMinutes": 0
}

# 函數：Base64編碼/解碼和HTTP請求
def toBase64(response):
    json_response = json.dumps(response, indent=4)
    encoded_string = base64.b64encode(json_response.encode())
    return encoded_string.decode()

def checkWeb(base64_encoded_string):
    global URL
    try:
        url = f"{URL}{base64_encoded_string}"
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

# 每秒更新計時器
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

# 定期檢查 detect flag，如果為 false 則關閉攝像頭
def check_detect_flag_periodically():
    global vs
    while True:
        time.sleep(20)  # 每 20 秒檢查一次
        print("[INFO] Checking detect flag...")
        requestData = {"MethodName": "GetDetectFlag"}
        request = toBase64(requestData)
        url = f"{URL}{request}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            decoded_response = original(response.text)
            print(decoded_response)

            status = decoded_response.get("Status")
            detect_flag = decoded_response.get("Message")

            if status == "Success" and detect_flag.lower() == "false":
                print("[INFO] DetectFlag is false. Stopping the video stream...")
                if vs is not None:
                    vs.release()  # 關閉攝像頭
                cv2.destroyAllWindows()
                sys.exit(0)  # 安全退出程序

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to check detect flag: {e}")

# EAR 和 MAR 計算
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

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

# 視訊處理主函數
def process_video_stream(vs, detector, predictor, lStart, lEnd, rStart, rEnd, mStart, mEnd):
    global ECOUNTER, ETOTAL, MCOUNTER, MTOTAL, elapsed_time_str, tempEtotal, tempMtotal, second

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

            if ear < EYE_EAR_THRESH:
                ECOUNTER += 1
            else:
                if ECOUNTER >= EYE_CONSEC_FRAMES:
                    ETOTAL += 1
                    tempEtotal += 1
                ECOUNTER = 0

            if mar > MOUTH_MAR_THRESH:
                MCOUNTER += 1
            else:
                if MCOUNTER >= MOUTH_CONSEC_FRAMES:
                    MTOTAL += 1
                    tempMtotal += 1
                MCOUNTER = 0

            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
            break

    vs.release()
    cv2.destroyAllWindows()

# 主程式邏輯
if __name__ == "__main__":
    # 加載偵測器和預測器
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    FACIAL_LANDMARK_68_IDXS = OrderedDict([
        ("mouth", (48, 68)), ("right_eye", (36, 42)), ("left_eye", (42, 48))
    ])

    (lStart, lEnd) = FACIAL_LANDMARK_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARK_68_IDXS["right_eye"]
    (mStart, mEnd) = FACIAL_LANDMARK_68_IDXS["mouth"]

    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    time.sleep(1.0)

    # 啟動計時器執行緒
    timer_thread = threading.Thread(target=time_start, daemon=True)
    timer_thread.start()

    # 啟動 detect flag 檢查執行緒
    detect_flag_thread = threading.Thread(target=check_detect_flag_periodically, daemon=True)
    detect_flag_thread.start()

    # 處理視訊流
    process_video_stream(vs, detector, predictor, lStart, lEnd, rStart, rEnd, mStart, mEnd)
