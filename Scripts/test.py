import pickle
scene_data_path = 'Runs\Boston\scene 1 res=0.5\Constant Power\data'
scene_data_path = scene_data_path.replace('\\', '/')
try:
    with open('.gitignore', 'r') as f:
        gitignore_contents = f.read().splitlines()
        for line in gitignore_contents:
            print(line)
            print(scene_data_path)
            print((line == scene_data_path))
    # Check if the data file is already in the .gitignore
    if any(scene_data_path in line for line in gitignore_contents):
        print(f"DATA FILE {scene_data_path} IS ALREADY IN THE .GITIGNORE FILE.")
    else:
        print(f'\nDATA FILE {scene_data_path} \nIS TOO BIG FOR GITHUB: ADD IT TO THE GITIGNORE FILE')
except:
    print()