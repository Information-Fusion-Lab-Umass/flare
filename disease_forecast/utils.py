import os

def create_dirs(paths):
    for path in paths:
        if os.path.exists(path)==False:
            os.mkdir(path)
