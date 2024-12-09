import face_recognition
import cv2
import pickle

def load_encodings(encoding_file='encodings.pkl'):
    """
    Load face encodings and names from a file.
    
    Args:
        encoding_file (str): Path to the file containing face encodings.
    
    Returns:
        dict: Dictionary containing encodings and names.
    """
    print("[INFO] Loading face encodings...")
    with open(encoding_file, "rb") as file:
        data = pickle.load(file)
    print("[INFO] Encodings loaded successfully!")
    return data

def real_time_face_recognition(encoding_file='encodings.pkl'):
    """
    Perform real-time face recognition using webcam and face encodings.

    Args:
        encoding_file (str): Path to the file containing face encodings.
    """
    # Load known face encodings and names
    data = load_encodings(encoding_file)
    known_encodings = data['encodings']
    known_names = data['names']

    # Start webcam video capture
    print("[INFO] Starting real-time face recognition...")
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        # Convert the frame to RGB (face_recognition requires RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and compute face encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each detected face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare the face encoding with known encodings
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            # Use the best match
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = face_distances.argmin()

            if matches[best_match_index]:
                name = known_names[best_match_index]

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw the person's name
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the video frame
        cv2.imshow("Real-Time Face Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Exiting real-time face recognition.")
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_face_recognition(encoding_file='encodings.pkl')
