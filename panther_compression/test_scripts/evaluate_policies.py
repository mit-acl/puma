#!/usr/bin/env python3

import subprocess
import argparse
import os

def main():

    # current directory
    cwd = os.getcwd()

    # get machine name
    machine = cwd.split("/")[2] 
    
    encoders = ["mlp", "lstm", "transformer", "gnn"]
    decoders = ["mlp", "diffusion"]

    for encoder in encoders:
        for decoder in decoders:
            model_path = get_model(encoder, decoder)
            print(model_path)
            cmd = f"python test_gnn_diffusion_training.py -en {encoder} -de {decoder} -m {machine} --train-model False --evaluate-after-training True --model-path {model_path}"
            subprocess.call(cmd.split(), cwd=cwd)

def get_model(encoder, decoder):
    model_path = f'/media/jtorde/T7/gdp/models/{encoder}_{decoder}/'
    # get the latest (creted time) model
    models = os.listdir(model_path)
    models.sort(key=lambda x: os.path.getctime(model_path + x))
    model = models[-1]
    model_path = model_path + model
    return model_path

if __name__ == "__main__":
    main()