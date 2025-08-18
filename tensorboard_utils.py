import subprocess
import os

def start_tensorboard(logdir='logs'):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    subprocess.Popen(['tensorboard', '--logdir', logdir, '--port', '6006'])
