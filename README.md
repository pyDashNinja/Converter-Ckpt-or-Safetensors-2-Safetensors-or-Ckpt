## Converter Ckpt 2 Safetensors & Safetensors 2 Ckpt

Special thanks to this repo : https://github.com/diStyApps/Safe-and-Stable-Ckpt2Safetensors-Conversion-Tool-GUI.git, All the credit goes to them.. I have just made simpler if someone wants to use only terminal to convert safetensors to ckpt and ckpt to safetensors without using gui and installing multiple libraries

```
pip install torch safetensors argparse
``` 

And Just Run..

```
Python converter.py --file_path="/path/to/file/{filename}.safetensors" --type_format="ckpt" --suffix="ckpt"
``` 


