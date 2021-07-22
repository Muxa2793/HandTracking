import cv2
import time
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)

folderPath = 'FingerImages'
myList = sorted(os.listdir(folderPath))
overlayList = []
pTime = 0

for imgPath in myList[1:]:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

detector = htm.handDetector(detectionConf=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, drawB=False, drawC=False, drawId=False)[0]

    if len(lmList) != 0:
        fingers = []

        # Большой палец
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Четыре пальца
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFinger = fingers.count(1)

        h, w, c = overlayList[totalFinger].shape
        img[0:h, 0:w] = overlayList[totalFinger]

        cv2.rectangle(img, (20, 400), (170, 600), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, str(totalFinger), (45, 550), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 10)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (1100, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow('Finger Count', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
