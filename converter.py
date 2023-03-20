import os
import torch 
from threading import Thread
from datetime import datetime as dt
from safetensors.torch import save_file
from safetensors.torch import load_file
from util.file_hash import get_file_hash
from constant import *
import argparse


format_file={
    "ckpt2safetensors": "ck2st",
    "safetensors2ckpt": "st2ck",
}


def process_file(file_path,type_format,suffix):
    if type_format == SAFETENSORS_STR:
        convert_to_st(file_path,suffix)
    if type_format == CKPT_STR:
        convert_to_ckpt(file_path,suffix)

def convert_to_st(checkpoint_path,suffix):
    model_hash = get_file_hash(checkpoint_path)
    weights = load_weights(checkpoint_path)
    file_name = f"{os.path.splitext(checkpoint_path)[0]}-cnvrtd.{SAFETENSORS_STR}" if suffix else f"{os.path.splitext(checkpoint_path)[0]}.{SAFETENSORS_STR}"
    save_file(weights, file_name)
    print(f'{CONVERTING_TXT} {checkpoint_path} [{model_hash}] to {SAFETENSORS_STR}.')
    print(f'Saving {file_name} [{get_file_hash(file_name)}].')

def convert_to_ckpt(filename,suffix):
    model_hash = get_file_hash(filename)
    device = "cpu"
    weights = load_file(filename, device=device)

    try:
        weights = load_file(filename, device=device)
    except Exception as e:
        print(f'Error: {e}')

    try:
        print(f'{CONVERTING_TXT} {filename} [{model_hash}] to {CKPT_STR}.')
        weights = load_file(filename, device=device)
        checkpoint_filename = f"{os.path.splitext(filename)[0]}-cnvrtd.{CKPT_STR}" if suffix else f"{os.path.splitext(filename)[0]}.{CKPT_STR}"
        save_checkpoint(weights, checkpoint_filename)
        print(f'Saving {checkpoint_filename} [{get_file_hash(checkpoint_filename)}].')
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            print(" File Not Found")
        else:
            print(f'Error: {e}')

def save_checkpoint(weights, filename):
    with open(filename, "wb") as f:
        torch.save(weights, f)

def load_weights(checkpoint_path):
    try:
        # Load the weights from the checkpoint file, without computing gradients
        with torch.no_grad():
            weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            # Check if the weights are contained in a "state_dict" key
            if "state_dict" in weights:
                weights = weights["state_dict"]
                # If the weights are nested in another "state_dict" key, remove it
                if "state_dict" in weights:
                    weights.pop("state_dict")
            return weights
            
    except Exception as e:
        pass

parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=str, help="Path to the file '.ckpt' or '.safetensors' : file to be converted")
parser.add_argument("type_format", type=str, help="type of format, converting into ckpt use 'ckpt' for safetensors use 'safetensors'")
parser.add_argument("suffix", type=str, help="its the suffix of the output file for converting into ckpt use 'ckpt' or for safetensors use 'safetensors'")
args = parser.parse_args()

process_file(file_path=args.file_path,type_format=args.type_format,suffix=args.suffix)
print("Done")


