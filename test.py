import os

def get_directory_tree(startpath='.'):
    """
    Prints the directory tree for the given path, skipping common virtual environments
    and other non-essential directories.
    
    :param startpath: The starting directory. Defaults to the current directory.
    """
    # List of directories to skip
    skip_dirs = ['.venv', '.git', '__pycache__', 'node_modules']
    
    for root, dirs, files in os.walk(startpath):
        # Modify the list of directories in-place to skip unwanted ones
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        # Calculate the depth of the current directory
        level = root.replace(startpath, '').count(os.sep)
        
        # Determine the indentation based on the level
        indent = ' ' * 4 * (level)
        
        # Print the current directory name
        print(f'{indent}{os.path.basename(root)}/')
        
        # Determine the indentation for files
        subindent = ' ' * 4 * (level + 1)
        
        # Print the files in the current directory
        for f in files:
            print(f'{subindent}{f}')

# Example usage:
# To print the tree of the current directory, skipping common directories
print("Directory Tree:")
get_directory_tree('.')