
from nicegui import ui

import auto_shutdown
from logger import logger
from datetime import datetime
import subprocess
import sys
import os
import signal
import psutil
from typing import Generator
import toml  # 用于保存和加载设置
import asyncio
import time
from threading import Thread
sys.path.append(os.path.join(os.path.dirname(__file__), 'musubi-tuner'))


# 输出绑定变量
preCacheLogger = None
trainLogger = None
settings_text = {'content': ''}


QWEN_IMAGE_SETTINGS_FILE = 'qwen_image_settings.toml'
# 预缓存进程
cache_process = None
cache_process_is_running = False
# 训练进程
train_process = None
train_process_is_running = False

def load_settings() -> dict:
    if os.path.exists(QWEN_IMAGE_SETTINGS_FILE):
        try:
            with open(QWEN_IMAGE_SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = toml.load(f)
                return settings
        except Exception:
            return {}
    else:
        return {}

def save_settings():
    try:
        with open(QWEN_IMAGE_SETTINGS_FILE, "w", encoding="utf-8") as f:
            toml.dump(qwen_image_training_settings, f)
    except Exception as e:
        print(f"[WARN] 保存 settings.toml 失败: {e}")
def bind_setting(ui_element, key):
    """将 UI 控件的值绑定到 qwen_image_training_settings[key] 并自动保存"""
    ui_element.on('update:model-value', lambda e: update_setting(key, e))

def update_setting(key, e):
    print(key,e.args)
    """通用更新方法，支持 input / checkbox / select 等"""
    value = e.args  # 原始值

    # 1. Checkbox 情况（[True, {...}]）
    if isinstance(value, list) and len(value) > 0:
        value = value[0]

    # 2. Select 情况（{'value': 1, 'label': 'xxx'}）
    elif isinstance(value, dict) and 'value' in value:
        value = value['label']

    # 3. 其余情况（input、slider、number 等），直接用 value

    qwen_image_training_settings[key] = value
    save_settings()
    preview_settings()

def preview_settings():



    toml_str = toml.dumps(qwen_image_training_settings)
    global settings_text
    settings_text.update(content = toml_str)




def writePreCacheLog(message):
    try:
        global preCacheLogger
        print('writePreCacheLog', 'message:', message,'EEEnd')
        if preCacheLogger:
            preCacheLogger.push(datetime.now().strftime("%Y-%m-%d %H:%M:%S ") +  message, classes='text-orange')

    except Exception as e:
        logger.info('logger error')
    logger.info(message)
def writeTrainLog(message):
    try:
        global trainLogger
        print('writeTrainLog', 'message:', message,'EEEnd')
        if trainLogger:
            trainLogger.push(datetime.now().strftime("%Y-%m-%d %H:%M:%S ") + message, classes='text-orange')
    except Exception as e:
        logger.info('logger error')
    logger.info(message)

qwen_image_training_settings = load_settings()
preview_settings()


def start_pre_caching():
    writePreCacheLog('开始执行预缓存...')
    """ 启动训练任务 """
    global cache_process

    dataset_config = qwen_image_training_settings['dataset_config']
    vae_path = qwen_image_training_settings['vae_path']
    text_encoder_model_path = qwen_image_training_settings['text_encoder_model_path']
    fp8 = qwen_image_training_settings['fp8']
    batch_size = qwen_image_training_settings['batch_size']
    edit = qwen_image_training_settings['edit']

    python_executable = sys.executable
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 拼接 musubi-tuner 目录下的脚本路径
    MUSUBI_DIR = os.path.join(base_dir, 'musubi-tuner','src','musubi_tuner')
    cache_latents_path = os.path.join(MUSUBI_DIR, "qwen_image_cache_latents.py")
    cache_text_encoder_path = os.path.join(MUSUBI_DIR, "qwen_image_cache_text_encoder_outputs.py")
    print(python_executable, cache_latents_path, cache_text_encoder_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        os.path.dirname(MUSUBI_DIR),  # LoRAMaster 根目录
        env.get("PYTHONPATH", "")
    ])
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LOG_LEVEL'] = 'DEBUG'

    cache_latents_cmd = [
        python_executable, cache_latents_path,
        "--dataset_config", dataset_config,
        "--vae", vae_path
    ]


    cache_text_encoder_cmd = [
        python_executable, cache_text_encoder_path,
        "--dataset_config", dataset_config,
        "--text_encoder", text_encoder_model_path,
        "--batch_size", batch_size
    ]
    if fp8:
        cache_text_encoder_cmd.append('--fp8_vl')
    if edit:
        cache_text_encoder_cmd.append('--edit')

    # 异步执行训练
    def run_cache():
        writePreCacheLog('开始执行预缓存 1/2 ...')
        writePreCacheLog(' '.join(cache_latents_cmd))
        global cache_process
        cache_process = subprocess.Popen(cache_latents_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

        for line in cache_process.stdout:
            writePreCacheLog(line.strip())

        return_code = cache_process.wait()
        cache_process = None
        if return_code != 0:
            writePreCacheLog(f"\n[ERROR] 命令执行失败，返回码: {return_code}\n")
        writePreCacheLog('预缓存 1/2 完成!')

        writePreCacheLog('开始执行预缓存2/2...')
        writePreCacheLog(' '.join(cache_text_encoder_cmd))
        cache_process = subprocess.Popen(cache_text_encoder_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         env=env)

        for line in cache_process.stdout:
            print(line)
            writePreCacheLog(line.strip())

        return_code = cache_process.wait()
        cache_process = None
        if return_code != 0:
            writePreCacheLog(f"\n[ERROR] 命令执行失败，返回码: {return_code}\n")
        writePreCacheLog('预缓存 2/2 完成!')



    Thread(target=run_cache).start()




def stop_pre_caching():
    """停止任务（绑定到停止按钮）"""
    global cache_process_is_running
    if not cache_process_is_running:
        return
    if cache_process:
        cache_process.terminate()
    cache_process_is_running = False
    writePreCacheLog('已停止预缓存!')


def terminate_process_tree(proc: subprocess.Popen):
    """
    递归终止指定进程及其所有子进程，适用于加速器或多进程场景。
    """
    if proc is None:
        return
    try:
        parent_pid = proc.pid
        if parent_pid is None:
            return
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        print(f"[WARN] terminate_process_tree 出现异常: {e}")

def stop_caching():
    msg = ''
    global  cache_process
    if cache_process is not None:
        proc = cache_process
        if proc.poll() is None:
            terminate_process_tree(proc)
            cache_process = None
            msg = "预缓存进程已被手动终止..."
        else:
            msg = "预缓存进程已经结束，无需停止..."
    else:
        msg = "当前没有正在进行的预缓存进程..."
    writePreCacheLog(msg)

def make_prompt_file(
    prompt_text: str,
    w: int,
    h: int,
    seed: int,
    steps: int,
    custom_prompt_txt: bool,
    custom_prompt_path: str,
    prompt_file_upload: str = None,
    image_path:str = None
) -> str:
    """
    如果上传了 prompt_file.txt，则直接返回该文件路径；
    否则，如果勾选了自定义且输入了路径，则返回该路径；
    否则自动生成默认的 prompt 文件。
    """
    if prompt_file_upload and os.path.isfile(prompt_file_upload):
        return prompt_file_upload
    elif custom_prompt_txt and custom_prompt_path.strip():
        return custom_prompt_path.strip()
    else:
        default_prompt_path = "./qwen_image_prompt_file.txt"
        with open(default_prompt_path, "w", encoding="utf-8") as f:
            f.write("# prompt 1: for generating a cat video\n")
            line = f"{prompt_text} --w {w} --h {h} --d {seed} --s {steps}"
            if image_path:
                line = line + ' --ci ' + image_path

            line = line + '\n'
            f.write(line)

        return default_prompt_path

def run_wan_training():
    dataset_config = qwen_image_training_settings['dataset_config']
    dit_weights_path = qwen_image_training_settings['dit_weights_path']
    vae_path = qwen_image_training_settings['vae_path']
    text_encoder_model_path = qwen_image_training_settings['text_encoder_model_path']
    edit = qwen_image_training_settings['edit']

    learning_rate = qwen_image_training_settings['learning_rate']
    gradient_accumulation_steps = qwen_image_training_settings['gradient_accumulation_steps']
    network_dim = qwen_image_training_settings['network_dim']
    timestep_sampling = qwen_image_training_settings['timestep_sampling']
    discrete_flow_shift = qwen_image_training_settings['discrete_flow_shift']
    max_train_epochs = qwen_image_training_settings['max_train_epochs']
    save_every_n_epochs = qwen_image_training_settings['save_every_n_epochs']
    save_every_n_steps = qwen_image_training_settings['save_every_n_steps']
    output_dir = qwen_image_training_settings['output_dir']
    output_name = qwen_image_training_settings['output_name']
    enable_low_vram = qwen_image_training_settings['enable_low_vram']
    blocks_to_swap = qwen_image_training_settings['blocks_to_swap']
    generate_samples = qwen_image_training_settings['generate_samples']
    sample_prompt_text = qwen_image_training_settings['sample_prompt_text']
    sample_image_path = qwen_image_training_settings['sample_image_path']
    sample_w = qwen_image_training_settings['sample_w']
    sample_h = qwen_image_training_settings['sample_h']
    sample_seed = qwen_image_training_settings['sample_seed']
    sample_steps = qwen_image_training_settings['sample_steps']
    custom_prompt_txt = qwen_image_training_settings['custom_prompt_txt']
    custom_prompt_path = qwen_image_training_settings['custom_prompt_path']
    sample_every_n_epochs = qwen_image_training_settings['sample_every_n_epochs']
    sample_every_n_steps = qwen_image_training_settings['sample_every_n_steps']
    sample_vae_path = qwen_image_training_settings['vae_path']
    fp8 = qwen_image_training_settings['fp8']
    num_cpu_threads_per_process = qwen_image_training_settings['num_cpu_threads_per_process']
    num_processes = qwen_image_training_settings['num_processes']
    attention_implementation = qwen_image_training_settings['attention_implementation']
    optimizer_type = qwen_image_training_settings['optimizer_type']
    max_data_loader_n_workers = qwen_image_training_settings['max_data_loader_n_workers']
    log_type = qwen_image_training_settings['log_type']
    log_prefix = qwen_image_training_settings['log_prefix']
    log_dir = qwen_image_training_settings['log_dir']
    log_tracker_name = qwen_image_training_settings['log_tracker_name']
    offload_inactive_dit = qwen_image_training_settings['offload_inactive_dit']
    mixed_precision = qwen_image_training_settings['mixed_precision']
    sample_at_first = qwen_image_training_settings['sample_at_first']



    python_executable = sys.executable
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 拼接 musubi-tuner 目录下的脚本路径
    MUSUBI_DIR = os.path.join(base_dir, 'musubi-tuner','src','musubi_tuner')
    train_network_path = os.path.join(MUSUBI_DIR, "qwen_image_train_network.py")
    print(python_executable, train_network_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        os.path.dirname(MUSUBI_DIR),  # LoRAMaster 根目录
        env.get("PYTHONPATH", "")
    ])
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LOG_LEVEL'] = 'DEBUG'


    command = [
        python_executable, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", str(num_cpu_threads_per_process),
        "--mixed_precision", mixed_precision,
        "--num_processes", str(num_processes),     # 只使用一个进程
        "--gpu_ids", "0",           # 只使用第一张GPU
        train_network_path,
        "--dit", dit_weights_path,
        "--dataset_config", dataset_config,
        "--vae",vae_path,
        "--text_encoder",text_encoder_model_path,
        "--mixed_precision", mixed_precision,
        "--optimizer_type", optimizer_type,
        "--learning_rate", learning_rate,
        "--gradient_checkpointing",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        "--max_data_loader_n_workers", str(max_data_loader_n_workers),
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora_qwen_image",
        "--network_dim", str(network_dim),
        "--timestep_sampling", timestep_sampling,
        "--discrete_flow_shift", str(discrete_flow_shift),
        "--max_train_epochs", str(max_train_epochs),
        "--save_every_n_epochs", str(save_every_n_epochs),
        "--save_every_n_steps", str(save_every_n_steps),
        "--output_dir", output_dir,
        "--output_name", output_name,
        "--seed","42",
        "--log_with",log_type
    ]
    if attention_implementation == 'sdpa':
        command.extend(['--sdpa'])
    elif attention_implementation == 'xformers':
        command.extend(['--xformers','--split_attn'])
    if offload_inactive_dit:
        command.extend(['--offload_inactive_dit'])
    if enable_low_vram:
        command.extend(["--blocks_to_swap", str(blocks_to_swap)])

    if fp8:
        command.extend(['--fp8_base'])
    if edit:
        command.extend(['--edit'])

    if generate_samples:
        prompt_file_final = make_prompt_file(
            prompt_text=sample_prompt_text,
            w=sample_w,
            h=sample_h,
            seed=sample_seed,
            steps=sample_steps,
            custom_prompt_txt=custom_prompt_txt,
            custom_prompt_path=custom_prompt_path,
            image_path=sample_image_path
        )
        command.extend([
            "--sample_prompts", prompt_file_final,
            "--sample_every_n_epochs", str(sample_every_n_epochs),
            "--sample_every_n_steps", str(sample_every_n_steps),
            "--vae", sample_vae_path,
            # "--fp8_llm"
        ])

        if sample_at_first:
            command.extend(["--sample_at_first"])

    if log_dir:
        command.extend(['--logging_dir',log_dir])
    if log_prefix:
        command.extend(['--log_prefix', log_prefix])
    if log_tracker_name:
        command.extend(['--log_tracker_name', log_tracker_name])


    def run_and_stream_output():
        global train_process
        train_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,env=env)

        for line in train_process.stdout:
            writeTrainLog(line)
        return_code = train_process.wait()
        train_process = None
        if return_code != 0:
            writeTrainLog(f"\n[ERROR] 命令执行失败，返回码: {return_code}\n")
        else:
            try:
                writeTrainLog('训练完成!')
            except Exception as e:
                print(f"{e}")
            # 自动关机
            if qwen_image_training_settings['auto_shutdown']:
                auto_shutdown.shutdown()


    writeTrainLog("开始运行 Wan LoRA训练命令...\n\n")
    writeTrainLog(' '.join(command))
    ui.notify("开始训练，完成前请不要离开本页面！",type='warning')
    Thread(target=run_and_stream_output).start()
def stop_train():
    global train_process

    if train_process is not None:
        proc = train_process
        if proc.poll() is None:
            terminate_process_tree(proc)
            train_process = None
            msg = "模型训练进程已被手动终止..."
        else:
            msg = "模型训练进程已经结束，无需停止..."
    else:
        msg = '当前没有正在进行的模型训练进程...'
    writeTrainLog(msg)

def draw_ui():
    ui.label('Qwen Image LoRA训练 (支持Qwen Image、Qwen Image Edit)').classes('text-2xl font-bold')
    with ui.row().classes('w-full no-wrap gap-4'):
        with ui.column().classes('w-2/3'):
            with ui.list().props('bordered separator').classes('w-full'):
                ui.item_label('① 基本设置').props('header').classes('text-xl font-bold mb-2')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Qwen_Image Edit / 训练千问Edit')
                        ui.item_label('打开后，训练qwen_image_edit，需要填写qwen_image_edit_bf16.safetensors').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        enable_low_vram = ui.switch(value=qwen_image_training_settings['edit']).props('outlined')
                        bind_setting(enable_low_vram, 'edit')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('DiT Weights Path / DiT权重文件路径')
                        ui.item_label('千问或千问edit的底膜，qwen_image_bf16.safetensors/qwen_image_edit_bf16.safetensors').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        dit_weights_path = ui.input(
                            placeholder='如I:\\train_models\\qwen_image\\qwen_image_bf16.safetensors',
                            value=qwen_image_training_settings['dit_weights_path']).props(
                            'rounded outlined dense').classes('w-full')
                        bind_setting(dit_weights_path, 'dit_weights_path')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Input toml path / 输入toml文件路径')
                        ui.item_label('toml配置文件的绝对路径，如如:E:\\qwen_image_lora_train\\config.toml').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        dataset_config = ui.input(placeholder='如:E:\\qwen_image_lora_train\\config.toml',
                                                  value=qwen_image_training_settings["dataset_config"]).classes(
                            'w-full').props('rounded outlined dense')
                        bind_setting(dataset_config, 'dataset_config')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('VAE File Path / Qwen_Image VAE文件路径')
                        ui.item_label(' Wan VAE文件的绝对路径，如: I:\\train_models\\qwen_image\\qwen_image_vae.safetensors').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        vae_path = ui.input(placeholder='如:  I:\\train_models\\qwen_image\\qwen_image_vae.safetensors',
                                            value=qwen_image_training_settings["vae_path"]).classes('w-full').props('rounded outlined dense')
                        bind_setting(vae_path, 'vae_path')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Text Encoder Model Path / Text Encoder模型路径')
                        ui.item_label('Text Encoder模型文件的绝对路径，如: I:\\train_models\\qwen_image\\qwen_2.5_vl_7b.safetensors').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        text_encoder_model_path = ui.input(placeholder='如: I:\\train_models\\qwen_image\\qwen_2.5_vl_7b.safetensors',
                                           value=qwen_image_training_settings["text_encoder_model_path"]).classes('w-full').props('rounded outlined dense')
                        bind_setting(text_encoder_model_path, 'text_encoder_model_path')


                with ui.item():
                    with ui.item_section():
                        ui.item_label('FP8')
                        ui.item_label('开启FP8模式，节省显存。针对于显存小于16GB的，建议勾选！').props('caption')
                    with ui.item_section().props('side'):

                        fp8 = ui.switch(value=qwen_image_training_settings['fp8'])
                        bind_setting(fp8, 'fp8')

            with ui.list().props('bordered separator').classes('w-full'):
                ui.item_label('② 预缓存').props('header').classes('text-xl font-bold mb-2')
                ui.separator()

                with ui.item():
                    with ui.item_section():
                        ui.item_label('Batch size / 批量大小')
                        ui.item_label('数值越大，计算越快，耗显存和内存越大').props('caption')
                    with ui.item_section().props('side'):
                        batch_size = ui.number(placeholder='同时送入 batch_size 条文本样本到 T5 编码器',
                                               value=qwen_image_training_settings["batch_size"]).style('width:200px').props('rounded outlined dense')
                        bind_setting(batch_size, 'batch_size')

                with ui.item():
                    with ui.row().classes('w-full no-wrap gap-4'):
                        ui.button('Run Pre-caching / 运行预缓存', on_click=start_pre_caching).classes('w-1/2')
                        ui.button('Stop Pre-caching / 停止预缓存', color='red', on_click=stop_caching).classes('w-1/2')

                with ui.item():
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label('输出日志').classes('text-xl font-bold')
                        ui.button('清空日志', on_click=lambda: preCacheLogger.clear())
                with ui.row().classes('w-full'):
                    global preCacheLogger
                    preCacheLogger = ui.log().classes('w-full h-30')


            with ui.list().props('bordered separator').classes('w-full'):
                ui.item_label('③ 正式训练').props('header').classes('text-xl font-bold mb-2')
                ui.separator()

                ui.separator()
                ui.label('训练过程：').classes('font-bold mb-2').style('margin-left:10px;margin-top:10px')
                ui.separator()

                # 训练基本参数
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Max Train Epochs / 最大训练轮数')
                        ui.item_label('>=2').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        max_train_epochs = ui.number(value=qwen_image_training_settings['max_train_epochs']).props('rounded outlined dense').classes('w-1/2')
                        bind_setting(max_train_epochs, 'max_train_epochs')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Learning Rate / 学习率')
                        ui.item_label('e.g. 2e-4').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        learning_rate = ui.input(value=qwen_image_training_settings['learning_rate']).props('rounded outlined dense').classes('w-1/2')
                        bind_setting(learning_rate, 'learning_rate')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Optimizer Type / 优化器类型')
                        ui.item_label(
                            '不同类型会影响 收敛速度、显存占用和最终效果。系统默认adamw8bit').props(
                            'caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        optimizer_type = ui.select(
                            ['adamw8bit', 'adamw', 'lion'],
                            value=qwen_image_training_settings['optimizer_type']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(optimizer_type, 'optimizer_type')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Network Dim / 网络维度')
                        ui.item_label('2-128，常用 4~128，不是越大越好, 低dim可以降低显存占用').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        network_dim = ui.number(value=qwen_image_training_settings['network_dim']).props('rounded outlined dense').classes('w-1/2')
                        bind_setting(network_dim, 'network_dim')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Mixed Precision / 混合精度')
                        ui.item_label(
                            '系统默认bf16').props(
                            'caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        mixed_precision = ui.select(
                            ['fp16', 'bf16'],
                            value=qwen_image_training_settings['mixed_precision']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(mixed_precision, 'mixed_precision')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Gradient Accumulation Steps / 梯度累积步数')
                    with ui.item_section().props('side').classes('w-1/2'):
                        gradient_steps = ui.number(value=qwen_image_training_settings['gradient_accumulation_steps']).props('rounded outlined dense').classes('w-1/2')
                        bind_setting(gradient_steps, 'gradient_accumulation_steps')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Timestep Sampling / 时间步采样')
                    with ui.item_section().props('side').classes('w-1/2'):
                        timestep_sampling = ui.input(value=qwen_image_training_settings['timestep_sampling']).props('rounded outlined dense').classes('w-1/2')
                        bind_setting(timestep_sampling, 'timestep_sampling')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('Discrete Flow Shift / 离散流移位')
                        ui.item_label('建议配置：2.2，仅供参考').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        discrete_shift = ui.number(value=qwen_image_training_settings['discrete_flow_shift'], format='%.1f').props('rounded outlined dense').classes('w-1/2')
                        bind_setting(discrete_shift, 'discrete_flow_shift')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('max_data_loader_n_workers / 数据加载的最大工作线程数')
                        ui.item_label('建议2-4，如果CPU 核心多、内存大、磁盘快，可以试着调高，比如 4、8、16，能更充分利用硬件加速数据读取。').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        max_data_loader_n_workers = ui.number(value=qwen_image_training_settings['max_data_loader_n_workers']).props('rounded outlined dense').classes('w-1/2')
                        bind_setting(max_data_loader_n_workers, 'max_data_loader_n_workers')
                ui.separator()
                ui.label('显存优化：').classes('font-bold mb-2').style('margin-left:10px;margin-top:10px')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Attention Implementation / 注意力实现')
                        ui.item_label(
                            '建议使用sdpa，速度和优化程度都很好，稳定，不用额外安装库，使用xformers需要装库').props(
                            'caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        attention_implementation = ui.select(
                            ['sdpa', 'xformers'],
                            value=qwen_image_training_settings['attention_implementation']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(attention_implementation, 'attention_implementation')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('Enable Low VRAM Mode / 启用低显存模式')
                        ui.item_label('使用低显存模式，牺牲训练速度，换取使用更低的显存').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        enable_low_vram = ui.switch(value=qwen_image_training_settings['enable_low_vram']).props('outlined')
                        bind_setting(enable_low_vram, 'enable_low_vram')

                with ui.item().bind_visibility_from(enable_low_vram, 'value'):
                    with ui.item_section():
                        ui.item_label('Blocks to Swap / 交换块数')
                        ui.item_label(
                            '数值越大，显存占用越低，训练速度越慢。16：24GB，45：12GB，仅供参考').props(
                            'caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        blocks_to_swap = ui.number(value=qwen_image_training_settings['blocks_to_swap']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(blocks_to_swap, 'blocks_to_swap')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('num_cpu_threads_per_process / 每个进程的CPU线程数')
                        ui.item_label('每个训练进程，开启的CPU线程数').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        num_cpu_threads_per_process = ui.number(
                            value=qwen_image_training_settings['num_cpu_threads_per_process']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(num_cpu_threads_per_process, 'num_cpu_threads_per_process')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('num_processes / 进程数')
                        ui.item_label('训练时，开启的进程数').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        num_processes = ui.number(value=qwen_image_training_settings['num_processes']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(num_processes, 'num_processes')

                ui.separator()
                ui.label('过程采样：').classes('font-bold mb-2').style(
                    'margin-left:10px;margin-top:10px')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Generate Samples During Training / 训练期间生成示例')
                        ui.item_label('在训练期间生成采样示例，注意，这里会拖慢训练速度！').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        generate_samples = ui.switch(value=qwen_image_training_settings['generate_samples'])
                        bind_setting(generate_samples, 'generate_samples')

                with ui.item().bind_visibility_from(generate_samples, 'value'):
                    with ui.item_section():
                        ui.item_label('Sample at first / 训练前生成示例')
                        ui.item_label('在训练开始前，先根据提示词生成一个示例！').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        sample_at_first = ui.switch(value=qwen_image_training_settings['sample_at_first'])
                        bind_setting(sample_at_first, 'sample_at_first')

                with ui.item().bind_visibility_from(generate_samples, 'value'):
                    with ui.row().classes('w-full no-wrap gap-4'):
                        sample_epoch = ui.number('Sample Every N Epochs / 每N个轮次采样一次',
                                                 value=qwen_image_training_settings['sample_every_n_epochs']).props(
                            'outlined').classes('w-1/2')
                        sample_step = ui.number('Sample Every N Steps / 每N步采样一次',
                                                value=qwen_image_training_settings['sample_every_n_steps']).props(
                            'outlined').classes('w-1/2')
                        bind_setting(sample_epoch, 'sample_every_n_epochs')
                        bind_setting(sample_step, 'sample_every_n_steps')
                with ui.item().bind_visibility_from(generate_samples, 'value'):
                    with ui.row().classes('w-full no-wrap gap-4'):
                        sample_prompt = ui.input('Prompt Text / 提示词',
                                                 value=qwen_image_training_settings['sample_prompt_text']).props(
                            'outlined').classes('w-full')
                        bind_setting(sample_prompt, 'sample_prompt_text')
                with ui.item().bind_visibility_from(generate_samples, 'value'):
                    with ui.row().classes('w-full no-wrap gap-4'):
                        sample_image_path = ui.input('Image Path / 图片路径（训练Qwen Image Edit时填写）',
                                                     value=qwen_image_training_settings['sample_image_path']).props(
                            'outlined').classes('w-full')
                        bind_setting(sample_image_path, 'sample_image_path')
                with ui.item().bind_visibility_from(generate_samples, 'value'):
                    with ui.row().classes('w-full no-wrap gap-4'):
                        sample_w = ui.number('Width (w) / 宽度', value=qwen_image_training_settings['sample_w']).props(
                            'outlined').classes(
                            'w-1/2')
                        sample_h = ui.number('Height (h) / 高度', value=qwen_image_training_settings['sample_h']).props(
                            'outlined').classes(
                            'w-1/2')

                        bind_setting(sample_w, 'sample_w')
                        bind_setting(sample_h, 'sample_h')
                with ui.item().bind_visibility_from(generate_samples, 'value'):
                    with ui.row().classes('w-full no-wrap gap-4'):
                        sample_seed = ui.number('Seed (d) / 种子', value=qwen_image_training_settings['sample_seed']).props(
                            'outlined').classes('w-1/2')
                        sample_steps = ui.number('Steps (s) / 步数', value=qwen_image_training_settings['sample_steps']).props(
                            'outlined').classes('w-1/2')

                        bind_setting(sample_seed, 'sample_seed')
                        bind_setting(sample_steps, 'sample_steps')






                ui.separator()
                ui.label('输出：').classes('font-bold mb-2').style('margin-left:10px;margin-top:10px')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Output Directory / 输出目录')
                    with ui.item_section().props('side').classes('w-1/2'):
                        output_dir = ui.input(value=qwen_image_training_settings['output_dir']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(output_dir, 'output_dir')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('Output Name / 输出名称')
                    with ui.item_section().props('side').classes('w-1/2'):
                        output_name = ui.input(value=qwen_image_training_settings['output_name']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(output_name, 'output_name')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('Save Every N Epochs / 每N个轮次保存一次')
                        ui.item_label('每执行N轮，就保存一次Lora模型').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        save_epochs = ui.number(value=qwen_image_training_settings['save_every_n_epochs']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(save_epochs, 'save_every_n_epochs')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Save Every N Steps / 每N步保存一次')
                        ui.item_label('每执行N步，就保存一次Lora模型').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        save_steps = ui.number(value=qwen_image_training_settings['save_every_n_steps']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(save_steps, 'save_every_n_steps')
                ui.separator()
                ui.label('日志：').classes('font-bold mb-2').style('margin-left:10px;margin-top:10px')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Log Type / 日志类型')
                    with ui.item_section().props('side').classes('w-1/2'):
                        log_type = ui.select(
                            ['tensorboard', 'wandb'],
                            value=qwen_image_training_settings['log_type']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(log_type, 'log_type')

                # with ui.item():
                #     with ui.item_section():
                #         ui.item_label('Log Prefix / 日志前缀')
                #     with ui.item_section().props('side').classes('w-1/2'):
                #         log_prefix = ui.input(value=qwen_image_training_settings['log_prefix']).props(
                #             'rounded outlined dense').classes('w-1/2')
                #         bind_setting(log_prefix, 'log_prefix')
                #
                # with ui.item():
                #     with ui.item_section():
                #         ui.item_label('Log Traker Name / 日志追踪器名字')
                #     with ui.item_section().props('side').classes('w-1/2'):
                #         log_tracker_name = ui.input(value=qwen_image_training_settings['log_tracker_name']).props(
                #             'rounded outlined dense').classes('w-1/2')
                #         bind_setting(log_tracker_name, 'log_traker_name')
                # with ui.item():
                #     with ui.item_section():
                #         ui.item_label('Log Dir / 日志目录')
                #         ui.item_label('日志保存位置').props('caption')
                #     with ui.item_section().props('side').classes('w-1/2'):
                #         log_dir = ui.input(value=qwen_image_training_settings['log_dir']).props(
                #             'rounded outlined dense').classes('w-1/2')
                #         bind_setting(log_dir, 'log_dir')
                ui.separator()
                ui.label('自动关机').classes('font-bold mb-2').style('margin-left:10px;margin-top:10px')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Auto Shutdown / 训练完成后自动关机')
                        ui.item_label('训练完成5分钟后，自动关机。注意：需要使用管理器启动才有效').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        aush = ui.switch(value=qwen_image_training_settings['auto_shutdown']).props('outlined')
                        bind_setting(aush, 'auto_shutdown')

                with ui.item():
                    with ui.row().classes('w-full no-wrap gap-4'):
                        ui.button('Run Training / 开始训练', on_click=run_wan_training).classes('w-1/2')
                        ui.button('Stop Training / 停止训练', color='red', on_click=stop_train).classes('w-1/2')

                with ui.item():
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label('输出日志').classes('text-xl font-bold')
                        ui.button('清空日志', on_click=lambda: trainLogger.clear())
                with ui.row().classes('w-full'):
                    global trainLogger
                    trainLogger = ui.log().classes('w-full h-30')


        with ui.column().classes('w-1/3').classes('bg-blue-100').style('padding:10px'):
            ui.label('参数预览').classes('text-xl font-bold mb-2')
            global settings_text
            settings_preview_label = ui.label('').bind_text_from(settings_text,'content').style("white-space: pre-wrap;")


