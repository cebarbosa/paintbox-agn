import os
import platform

node = platform.node()
lai_machines = ["uv100", "alphacrucis", "yaci.iag.usp.br"] # nodes @LAI cluster
# Settings for personal computer
if node in ["kadu-Inspiron-5557"]:
    home_dir = "/home/kadu/Dropbox/paintbox-agn"
    cvd_dir = "/home/kadu/Dropbox/SSPs/CvD18"
    mp_pool_size = 4 # Number of cores for parallel processing
# Settings for supercomputers @IAG/USP
elif node in lai_machines:
    home_dir = "/sto/home/cebarbosa/imacs-imf"
    cvd_dir = "/sto/home/cebarbosa/SSPs/CvD18"
    mp_pool_size = 64
else:
    raise ValueError("Please define your directories inside context.py")

data_dir = os.path.join(home_dir, "data")