import cv2
import time
from roboflow import Roboflow
from datetime import datetime

def setup_roboflow():
    rf = Roboflow(api_key="SrpFJK8EZKO4eKywSP4m")
    project = rf.workspace("stringing-fault-detection").project("stringing-faults")
    model = project.version(4).model
    return model

def save_frame(frame):
    # Save frame to a temporary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_filename = f"temp_frame_{timestamp}.jpg"
    cv2.imwrite(temp_filename, frame)
    return temp_filename

def main():
    # Initialize Roboflow
    model = setup_roboflow()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set webcam resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_prediction_time = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break

            # Display the frame
            cv2.imshow('Webcam Feed', frame)

            # Check if it's time for a new prediction (1 second interval)
            current_time = time.time()
            if current_time - last_prediction_time >= 0.2:
                # Save frame and perform prediction
                temp_filename = save_frame(frame)
                
                try:
                    # Perform prediction
                    prediction = model.predict(temp_filename, confidence=40, overlap=30).json()
                    print(f"\nPrediction at {datetime.now().strftime('%H:%M:%S')}:")
                    print(prediction)
                    
                except Exception as e:
                    print(f"Prediction error: {e}")

                # Update last prediction time
                last_prediction_time = current_time

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()