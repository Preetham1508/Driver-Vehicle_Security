from model import FaceNetModel
from cam_register import register_face
from cam_verify import verify_face

if __name__ == "__main__":
    model = FaceNetModel(emd_size=256)  # Load the model
    
    choice = input("Enter 'r' to register a face, 'v' to verify a face, or 'q' to quit: ").strip().lower()

    while choice != 'q':
        if choice == 'r':
            username = input("Enter username to register: ").strip()
            register_face(model, username)  # Register face for the given username
        elif choice == 'v':
            username = input("Enter username to verify: ").strip()
            verify_face(model, username)  # Verify face for the given username
        else:
            print("Invalid choice. Please enter 'r', 'v', or 'q'.")
        
        # Ask again for the user's choice
        choice = input("Enter 'r' to register a face, 'v' to verify a face, or 'q' to quit: ").strip().lower()

    print("Goodbye!")
