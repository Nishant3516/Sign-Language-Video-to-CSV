import cv2
from cvzone.PoseModule import PoseDetector

# Replace the video file path with your actual path
cap = cv2.VideoCapture('technician (sign 1).mp4')

detector = PoseDetector()
posList = []

while True:
    success, img = cap.read()

    # Check if the frame is successfully read
    if not success:
        break

    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)

    if bboxInfo:
        lmString = ''
        for lm in lmList:
            if len(lm) >= 4:  # Ensure that lm has at least 4 elements
                lmString += f'{lm[1]},{img.shape[0] - lm[2]},{lm[3]},'
        posList.append(lmString)

    print(len(posList))

    cv2.imshow('Pose Detection', img)  # Use cv2.imshow for VS Code
    key = cv2.waitKey(1)
    if key == ord('s'):
        with open("AnimationFile.txt", 'w') as f:
            f.writelines(["%s\n" % item for item in posList])

    if key == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
cap.release()
