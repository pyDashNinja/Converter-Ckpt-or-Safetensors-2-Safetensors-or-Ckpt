# import PySimpleGUI as sg
import os
import torch 
from threading import Thread
from datetime import datetime as dt
from safetensors.torch import save_file
from safetensors.torch import load_file
# from util.ui_flattener import flatten_ui_elements
from util.file_hash import get_file_hash
# import util.progress_bar_custom as cpbar
# import util.file_explorer_component as file_explorer
# import util.colors as color
# import util.support as support
from CONSTANT import *


format_file={
    "ckpt2safetensors": "ck2st",
    "safetensors2ckpt": "st2ck",
}


def process_file(file_path,type_format,suffix):
    if type_format == SAFETENSORS_STR:
        convert_to_st(file_path,suffix)
    if type_format == 'CKPT_STR':
        print("here")
        convert_to_ckpt(file_path,suffix)

    # cpbar.progress_bar_custom(0,1,start_time,window,PBAR_KEY,"file")

    # file_explorer_lstbox_key_elem.update(file_explorer.get_system_files(file_explorer.CurrentDirectory.path, sort="ASC"))
    # folder_browse_inp_elem.update(file_explorer.CurrentDirectory.path)
    # convert_button_enable()

    # file_explorer.SelectedFileSystem.path = None

# def process_directory(path,idx,type_format,suffix):
    # if type_format == CKPT_STR:
    #     _file_extensions = PYTORCH_FILE_EXTENSIONS
    # if type_format == SAFETENSORS_STR:
    #     _file_extensions = SAFETENSORS_FILE_EXTENSIONS

    # for base_path, _, file_names in os.walk(path):
    #     for file_name in file_names:
    #         file_ext = os.path.splitext(file_name)[1]
    #         if (
    #             file_ext not in _file_extensions
    #         ):
    #             continue
    #         file_path = os.path.join(base_path, file_name)
    #         input_directory_path_list.append(file_path)
    #         print(f' {file_name}')

    #     if type_format == CKPT_STR:
    #         for file_name in input_directory_path_list:
    #             idx = (idx+1)
    #             convert_to_st(file_name,suffix)
    #             cpbar.progress_bar_custom(idx-1,len(input_directory_path_list),start_time,window,PBAR_KEY,"file")

    #     if type_format == SAFETENSORS_STR:
    #         for file_name in input_directory_path_list:
    #             idx = (idx+1)
    #             convert_to_ckpt(file_name,suffix)
    #             cpbar.progress_bar_custom(idx-1,len(input_directory_path_list),start_time,window,PBAR_KEY,"file")
        
    # file_explorer_lstbox_key_elem.update(file_explorer.get_system_files(file_explorer.CurrentDirectory.path, sort="ASC"))
    # folder_browse_inp_elem.update(file_explorer.CurrentDirectory.path)
    # convert_button_enable()

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

process_file(file_path="../uberRealisticPornMerge_urpmv13.safetensors",type_format='CKPT_STR',suffix='ckpt')
print("jer")


