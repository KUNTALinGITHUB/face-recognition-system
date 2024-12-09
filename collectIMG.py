import cv2
import os

def collect_images(person_name, save_dir='dataset', num_images=500):
    """
    Collect images for a person and save them into a dedicated folder.

    Args:
        person_name (str): The name of the person (used for folder creation).
        save_dir (str): Directory where the images will be saved.
        num_images (int): Number of images to collect.
    """
    # Create a folder for the person
    person_dir = os.path.join(save_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)  # Start capturing video from the webcam
    count = 0

    print(f"\nCollecting {num_images} images for '{person_name}'. Press 'q' to stop early.")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing webcam. Exiting...")
            break

        # Show the video feed
        cv2.imshow("Collecting Images", frame)

        # Save the current frame as an image
        img_path = os.path.join(person_dir, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

        # Option to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Manual exit triggered.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Collected {count} images for '{person_name}'. Images saved in '{person_dir}'.")

# Main program: Loop to collect images for multiple individuals
def main():
    save_dir = 'dataset'
    num_images = 500  # Number of images per person

    while True:
        # Input for person's name
        person_name = input("\nEnter the person's name (or type 'exit' to quit): ").strip()
        
        if person_name.lower() == 'exit':
            print("Exiting program. Goodbye!")
            break

        # Start image collection for this person
        collect_images(person_name, save_dir, num_images)

if __name__ == "__main__":
    main()
