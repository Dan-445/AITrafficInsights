import cv2

# Callback function to capture the mouse click and display coordinates
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Display the coordinates on the console
        print(f"Coordinates: X={x}, Y={y}")
        # Display the coordinates on the image window
        cv2.putText(img_resized, f"({x},{y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Frame', img_resized)

# Capture video from file
cap = cv2.VideoCapture('2-sept/15106/2024_0830_092733_050A.MP4')

# Read the first frame from the video
ret, img = cap.read()

if ret:
    # Resize the frame to 480x640
    img_resized = cv2.resize(img, (640, 480))
    
    # Show the resized frame in a window
    cv2.imshow('Frame', img_resized)
    
    # Set the mouse callback function to the window
    cv2.setMouseCallback('Frame', click_event)

    # Wait for the user to press 'q' to exit
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
