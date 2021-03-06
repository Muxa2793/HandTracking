import os
import cv2
import time
import numpy as np
import math
import HandTrackingModule as htm


cap = cv2.VideoCapture(0)
pTime = 0

detector = htm.handDetector(detectionConf=0.7)

minVol = 0
maxVol = 100
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
     
    # Поиск руки
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, drawC=False, drawId=False)
    
    if len(lmList) != 0:

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = math.hypot(x2-x1, y2-y1)

        # Интерполяция предела смыкания и размыкания пальцев и максимальной и минимальной громкости
        vol = np.interp(length, [60, 300], [minVol, maxVol])
        volBar = vol = np.interp(length, [60, 300], [600, 400])
        volPer = vol = np.interp(length, [60, 300], [0, 100])

        if length <= 60:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        elif length >= 300:
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        else:
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        # Изменение уровня громкости
        os.system(f"volume {str(vol)}")

    # Расчёт FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f'{int(volPer)}%', (40, 630), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.rectangle(img, (40, 400), (80, 600), (255, 0, 0), 2)
    cv2.rectangle(img, (40, int(volBar)), (80, 600), (255, 0, 0), cv2.FILLED)

    cv2.imshow('Volume Control', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
