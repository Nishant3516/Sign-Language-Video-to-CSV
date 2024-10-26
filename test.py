import cv2
import json
from cvzone.PoseModule import PoseDetector

# Initialize video capture and pose detector
cap = cv2.VideoCapture('technician (sign 1).mp4')
detector = PoseDetector()
pose_data = []  # List to store pose data for all frames
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Apply pose detection on the current frame
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True)

    # Print lmList for debugging
    print(f"Frame {frame_count}, lmList: {lmList}")

    # Initialize data structure for each frame
    frame_data = {"frame": frame_count, "landmarks": {}}

    # Check if lmList contains landmark data
    if lmList:
        for i, lm in enumerate(lmList):
            # Make sure each landmark entry has x, y, z
            if len(lm) >= 3:
                frame_data["landmarks"][f"lm{i}"] = {
                    "x": lm[0],
                    "y": lm[1],
                    "z": lm[2]
                }
            else:
                frame_data["landmarks"][f"lm{i}"] = {
                    "x": None, "y": None, "z": None}
    else:
        # If no landmarks were found, add None for each expected landmark
        print(f"No landmarks found in frame {frame_count}")
        for i in range(33):  # Assuming 33 landmarks
            frame_data["landmarks"][f"lm{i}"] = {
                "x": None, "y": None, "z": None}

    # Append frame data to the main list
    pose_data.append(frame_data)
    frame_count += 1

    # Display pose detection output (optional)
    cv2.imshow('Pose Detection', img)
    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

# Write all collected pose data to a JSON file
with open("pose_data_3D.json", "w") as f:
    json.dump(pose_data, f, indent=4)

# Release resources
cv2.destroyAllWindows()
cap.release()
