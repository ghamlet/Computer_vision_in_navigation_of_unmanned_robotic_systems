from argparse import ArgumentParser
import os
from pathlib import Path


convert_tool_path = str(Path.home() / 'Tengine/build/install/bin/convert_tool'    )
quant_tool_path   = str(Path.home() / 'Tengine/build/install/bin/quant_tool_uint8')

parser = ArgumentParser()
parser.add_argument('--weights', '-w', type=str, required=True)
parser.add_argument('--config',  '-c',  type=str, required=True)
parser.add_argument('--dataset', '-d', type=str, required=True)

args = parser.parse_args()
weights_file = Path(args.weights).absolute()
config_file  = Path(args.config ).absolute()
dataset_dir  = Path(args.dataset).absolute()

if not weights_file.exists():
    raise FileNotFoundError('Weights file was not found!')

if not config_file.exists():
    raise FileNotFoundError('Config file was not found!')

if not dataset_dir.exists():
    raise FileNotFoundError('Dataset directory was not found!')

converted_model = weights_file.with_name("_converted_"+weights_file.name).with_suffix('.tmfile')
output_name = weights_file.name[:-len(weights_file.suffix)] + '_uint8.tmfile'
output_model = weights_file.with_name(output_name)

os.system(f'{convert_tool_path} -f darknet -p {config_file} -m {weights_file} -o {converted_model}')
os.system(f'{quant_tool_path} -m {converted_model} -o {output_model} -i {dataset_dir} -g 3,416,416 -w 0,0,0 -s 0.003921,0.003921,0.003921 -c 0 -y 416,416 -t 8')

os.remove(converted_model)
