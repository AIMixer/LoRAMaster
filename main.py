from nicegui import ui
from layout import with_layout  # 如果封装在 layout.py
import  wan_lora_train
import about
import framepack_lora_train
import  hunyuan_lora_train
import kontext_lora_train
import qwen_image_lora_train
import  sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), 'musubi-tuner'))
import subprocess
import atexit

# 启动 tensorboard（后台运行）
tensorboard_process = subprocess.Popen([
    'tensorboard',
    '--logdir', './logs',     # 你的训练日志路径
    '--port', '6006',
    '--host', '0.0.0.0'
])

@atexit.register
def cleanup():
    tensorboard_process.terminate()


@ui.page('/')
def dashboard_page():
    def content():
        about.draw_ui()
    with_layout(content)


@ui.page('/Wan')
def wan_train_page():
    def content():
        wan_lora_train.draw_ui()
    with_layout(content)

@ui.page('/FramePack')
def framepack_train_page():
    def content():
        framepack_lora_train.draw_ui()
    with_layout(content)

@ui.page('/HunyuanVideo')
def hunyuan_train_page():
    def content():
        hunyuan_lora_train.draw_ui()
    with_layout(content)

@ui.page('/FluxKontext')
def kontext_train_page():
    def content():
        kontext_lora_train.draw_ui()
    with_layout(content)

@ui.page('/QwenImage')
def qwen_image_page():
    def content():
        qwen_image_lora_train.draw_ui()
    with_layout(content)

@ui.page('/Tensorboard')
def settings_page():
    def content():
        ui.html(
            '<iframe src="http://127.0.0.1:6006" '
            'style="width:100%; height:100%; border:none;"></iframe>'
        ).style('width:100%;height:100vh;')
    with_layout(content)

# --------------------------
# 启动默认首页
# --------------------------
ui.run(reload=False,title='LoRA训练大师 - by AI搅拌手',port=8080)