#!/usr/bin/env python3

import subprocess
import os

# current directory
cwd = os.getcwd()

# get machine name
machine = cwd.split("/")[2] 

# run python script
encoders = ["mlp", "lstm", "transformer", "gnn"]
decoders = ["mlp", "diffusion"]

for encoder in encoders:
    for decoder in decoders:
        cmd = f"python test_gnn_diffusion_training.py -en {encoder} -de {decoder} -m {machine}"
        print(cmd.split())
        subprocess.call(cmd.split(), cwd=cwd)