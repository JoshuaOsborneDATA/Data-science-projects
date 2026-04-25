import os

if not os.path.isdir("../data/"):
    os.mkdir("../data/")
    os.mkdir("../data/raw/")
    
if not os.path.isdir("../models/"):
    os.mkdir("../models/")
    
if not os.path.isdir("../notebooks/"):
    os.mkdir("../notebooks/")