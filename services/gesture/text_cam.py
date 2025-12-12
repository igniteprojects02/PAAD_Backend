import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    print("Frame:", ret)   # ← tells us if camera is working

    if ret:
        cv2.imshow("Test Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
