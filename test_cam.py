import cv2

# Try opening the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ ERROR: CAMERA FAILED TO OPEN. Try changing 0 to 1.")
else:
    print("✅ CAMERA OPENED SUCCESSFULLY!")
    print("Press 'q' to quit the video window.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: CANNOT READ FRAME. Camera is being used by another app.")
            break
            
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()