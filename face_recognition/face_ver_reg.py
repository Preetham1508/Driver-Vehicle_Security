from model import FaceNetModel
from cam_register import register_face
from cam_verify import verify_face

if __name__ == "__main__":
    model = FaceNetModel(emd_size=256)
    choice = input("Enter 'r' to register a face, 'v' to verify a face, or 'q' to quit: ").strip().lower()

    while choice != 'q':
        if choice == 'r':
            register_face(model)
        elif choice == 'v':
            verify_face(model)
        else:
            print("Invalid choice. Please enter 'r', 'v', or 'q'.")
        choice = input("Enter 'r' to register a face, 'v' to verify a face, or 'q' to quit: ").strip().lower()

    print("Goodbye!")
