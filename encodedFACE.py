import face_recognition
import os
import cv2
import pickle

def encode_faces(dataset_dir='dataset', encoding_file='encodings.pkl'):
    """
    Encode faces from the images in the dataset directory and save the encodings.

    Args:
        dataset_dir (str): Directory containing subfolders of images per person.
        encoding_file (str): File to save the face encodings and names.
    """
    known_encodings = []  # List to store face encodings
    known_names = []      # List to store corresponding names

    print("\n[INFO] Start encoding faces...")
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)

        # Check if it is a directory
        if not os.path.isdir(person_dir):
            continue

        print(f"[INFO] Processing images for '{person_name}'...")
        
        # Loop through all images in the person's folder
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)

            # Load the image
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (face_recognition requirement)

            # Detect face locations and encodings
            boxes = face_recognition.face_locations(rgb_image, model='hog')  # 'hog' is faster; use 'cnn' for accuracy
            encodings = face_recognition.face_encodings(rgb_image, boxes)

            # Store the encodings and the person's name
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(person_name)

    # Save the encodings and names to a file
    print(f"\n[INFO] Saving encodings to '{encoding_file}'...")
    data = {"encodings": known_encodings, "names": known_names}
    with open(encoding_file, "wb") as file:
        pickle.dump(data, file)

    print(f"[INFO] Encoding complete! Encodings saved to '{encoding_file}'.")

if __name__ == "__main__":
    encode_faces(dataset_dir='dataset', encoding_file='encodings.pkl')
