import subprocess

def run_training(name, data_path, output_path, on_update=None):
    cmd = [
        'python', 'train_network.py',  # musubi-tuner脚本
        '--train_data_dir', data_path,
        '--output_dir', output_path,
        '--output_name', name,
        '--resolution', '512,512',
        '--network_module=networks.lora',
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line.strip())
        if on_update:
            on_update(line.strip())
