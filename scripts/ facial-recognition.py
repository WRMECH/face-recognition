import face_recognition
import numpy as np
import os

def recognize_faces(known_faces_dir, unknown_image_path):
    """
    Performs facial recognition on an unknown image against a directory of known faces.

    Args:
        known_faces_dir (str): Path to the directory containing known face images.
        unknown_image_path (str): Path to the image containing unknown faces.
    """
    known_face_encodings = []
    known_face_names = []

    # Load known faces and their encodings
    print(f"Loading known faces from: {known_faces_dir}")
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0] # Use filename as name
            image_path = os.path.join(known_faces_dir, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"  Loaded: {name}")
                else:
                    print(f"  No face found in {filename}. Skipping.")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")

    if not known_face_encodings:
        print("No known faces loaded. Please ensure the 'known_faces' directory contains images with faces.")
        return

    # Load the unknown image
    print(f"\nProcessing unknown image: {unknown_image_path}")
    try:
        unknown_image = face_recognition.load_image_file(unknown_image_path)
    except FileNotFoundError:
        print(f"Error: Unknown image file not found at {unknown_image_path}")
        return
    except Exception as e:
        print(f"Error loading unknown image: {e}")
        return

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    print(f"Found {len(face_locations)} face(s) in the unknown image.")

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        print(f"  Face at ({top}, {right}, {bottom}, {left}) is: {name}")

# --- How to use ---
# 1. Create a directory named 'known_faces' in the same location as this script.
# 2. Place images of people you want to recognize inside 'known_faces'.
#    Name the image files after the person (e.g., 'john_doe.jpg', 'jane_smith.png').
# 3. Replace 'path/to/your/unknown_image.jpg' with the actual path to the image you want to analyze.

if __name__ == "__main__":
    # Example usage:
    # Make sure to create a 'known_faces' directory and place images there.
    # Also, provide a path to an 'unknown_image.jpg' for testing.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    known_faces_directory = os.path.join(current_dir, "known_faces")
    unknown_image_to_test = os.path.join(current_dir, "unknown_image.jpg") # Replace with your actual unknown image path

    # Create dummy directories and files for demonstration if they don't exist
    if not os.path.exists(known_faces_directory):
        os.makedirs(known_faces_directory)
        print(f"Created dummy directory: {known_faces_directory}")
        print("Please add known face images (e.g., 'john.jpg') to this directory.")

    # You would typically have a real image here.
    # For demonstration, we'll just print a message if the unknown image is missing.
    if not os.path.exists(unknown_image_to_test):
        print(f"\nWarning: '{unknown_image_to_test}' not found. Please create or provide a path to your unknown image.")
        print("Example: You can place an image named 'unknown_image.jpg' in the same directory as this script.")
    else:
        recognize_faces(known_faces_directory, unknown_image_to_test)
