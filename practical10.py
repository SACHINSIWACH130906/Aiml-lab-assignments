import cv2

# Open CCTV video (place cctv.mp4 in the same folder as this file)
video = cv2.VideoCapture("cctv.mp4")

# Check if video opened successfully
if not video.isOpened():
    print("Error: Cannot open cctv.mp4")
    print("Make sure the video file is in the same folder as this script.")
    exit()

# Create background subtractor (MOG2)
detector = cv2.createBackgroundSubtractorMOG2()

print("Video opened successfully. Press 'q' to quit.")

while True:
    # Read video frame
    ret, frame = video.read()

    # If no frame is read (end of video), break
    if not ret:
        print("End of video.")
        break

    # Detect moving objects (foreground mask)
    mask = detector.apply(frame)

    # Find contours (boundaries of moving objects)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw bounding boxes around large contours
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),   # Green box
                2              # Thickness
            )

    # Display the frame
    cv2.imshow("CCTV Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
print("Done.")