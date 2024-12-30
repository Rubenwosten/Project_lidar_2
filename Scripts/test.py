import pickle

file_path = "Runs/Boston/scene 1 res=0.5/Constant Power/data"
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("File loaded successfully:", data)
except EOFError:
    print("EOFError: File is empty or corrupted.")
except Exception as e:
    print(f"An error occurred: {e}")