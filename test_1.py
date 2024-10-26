import cv2
import csv
from cvzone.PoseModule import PoseDetector

# Initialize video capture and pose detector
cap = cv2.VideoCapture('technician (sign 1).mp4')
detector = PoseDetector()
frame_count = 0

# CSV file setup
csv_file = "pose_data_3D.csv"
num_landmarks = 33  # Assuming 33 landmarks

# Initialize CSV header
header = ["frame"]
for i in range(num_landmarks):
    header.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])

# Open CSV file for writing
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header

    while True:
        success, img = cap.read()
        if not success:
            break

        # Apply pose detection on the current frame
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True)

        # Prepare row data for each frame
        row_data = [frame_count]

        # If landmarks are detected, populate them in row data; else, use None for missing landmarks
        if lmList:
            for lm in lmList:
                # Add x, y, z for each landmark
                row_data.extend([lm[0], lm[1], lm[2]])
            # If fewer than 33 landmarks are detected, pad with None
            if len(lmList) < num_landmarks:
                row_data.extend([None, None, None] *
                                (num_landmarks - len(lmList)))
        else:
            # If no landmarks detected, fill with None for all landmarks
            row_data.extend([None, None, None] * num_landmarks)

        # Write the frame's row data to CSV
        writer.writerow(row_data)
        frame_count += 1

        # Display pose detection output (optional)
        cv2.imshow('Pose Detection', img)
        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break

# Release resources
cv2.destroyAllWindows()
cap.release()
