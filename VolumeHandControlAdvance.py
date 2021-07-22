import cv2
import time
import numpy as np
import HandTrackingModule as htm
import osascript


def main():

    cap = cv2.VideoCapture(0)
    area = 0
    pTime = 0

    detector = htm.handDetector(detectionConf=0.8)

    volBar = 400
    volPer = 0
    cVol = get_volume_info()

    while True:
        success, img = cap.read()

        # Поиск руки
        img = detector.findHands(img, draw=True)
        lmList, bbox = detector.findPosition(img, drawC=False, drawId=False, drawB=True)

        if len(lmList) != 0:

            area = (bbox[2] - bbox[0])*(bbox[3] - bbox[1])//100

            if 250 < area < 2000:

                length, img, lineInfo = detector.findDistance(4, 8, img)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (255, 0, 0), cv2.FILLED)

                # Интерполяция предела смыкания и размыкания пальцев и максимальной и минимальной громкости

                volBar = np.interp(length, [60, 200], [600, 400])
                volPer = np.interp(length, [60, 200], [0, 100])

                # Шаг изменения громкости
                smoothness = 5  # Шаг 5
                volPer = smoothness * round(volPer/smoothness)

                # Проверка подняты пальцы или нет
                fingers = detector.fingersUp()

                # Изменение уровня громкости, если мизинец опущен
                if not fingers[3]:
                    osascript.osascript(f"set volume output volume {str(volPer)}")
                    cVol = get_volume_info()
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)

        # Расчёт FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f'{int(volPer)}%', (40, 630), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f'Current VOL: {int(cVol)}', (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv2.rectangle(img, (40, 400), (80, 600), (255, 0, 0), 2)
        cv2.rectangle(img, (40, int(volBar)), (80, 600), (255, 0, 0), cv2.FILLED)

        cv2.imshow('Volume Control', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_volume_info():
    result = osascript.osascript('get volume settings')
    volInfo = result[1].split(',')
    cVol = volInfo[0].replace('output volume:', '')
    return cVol


if __name__ == '__main__':
    main()
