import os
# If you named your folder 'data', use 'data'.
# Otherwise, put the exact name here.
data_path = 'data'

if os.path.exists(data_path):
    classes = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    print(f"Found {len(classes)} animal classes.\n")
    
    for animal in sorted(classes):
        path = os.path.join(data_path, animal)
        count = len(os.listdir(path))
        print(f"{animal:10} : {count} images")
else:
    print("Error: The folder path is incorrect!")