
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
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))


# 输出绑定变量
preCacheLogger = None
trainLogger = None
settings_text = {'content': ''}


FLUX_SETTINGS_FILE = 'flux_settings.toml'
# 预缓存进程
cache_process = None
cache_process_is_running = False
# 训练进程
train_process = None
train_process_is_running = False

def load_settings() -> dict:
    if os.path.exists(FLUX_SETTINGS_FILE):
        try:
            with open(FLUX_SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = toml.load(f)
                return settings
        except Exception:
            return {}
    else:
        return {}

def save_settings():
    try:
        with open(FLUX_SETTINGS_FILE, "w", encoding="utf-8") as f:
            toml.dump(flux_training_settings, f)
    except Exception as e:
        print(f"[WARN] 保存 settings.toml 失败: {e}")
def bind_setting(ui_element, key):
    """将 UI 控件的值绑定到 flux_training_settings[key] 并自动保存"""
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

    flux_training_settings[key] = value
    save_settings()
    preview_settings()

def preview_settings():

    toml_str = toml.dumps(flux_training_settings)
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

flux_training_settings = load_settings()
preview_settings()




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


def make_prompt_file(
    prompt_text: str,
) -> str:

    default_prompt_path = "./flux_prompt_file.txt"
    with open(default_prompt_path, "w", encoding="utf-8") as f:
        f.write("# prompt 1: for generating a cat video\n")
        line = f"{prompt_text}"
        line = line + '\n'
        f.write(line)

    return default_prompt_path

def run_wan_training():
    dataset_config = flux_training_settings['dataset_config']
    vae_path = flux_training_settings['vae_path']
    clip_model_path = flux_training_settings['clip_model_path']
    t5_path = flux_training_settings['t5_path']
    dit_weights_path = flux_training_settings['dit_weights_path']
    learning_rate = flux_training_settings['learning_rate']
    gradient_accumulation_steps = flux_training_settings['gradient_accumulation_steps']
    network_dim = flux_training_settings['network_dim']
    timestep_sampling = flux_training_settings['timestep_sampling']
    max_train_epochs = flux_training_settings['max_train_epochs']
    save_every_n_epochs = flux_training_settings['save_every_n_epochs']
    save_every_n_steps = flux_training_settings['save_every_n_steps']
    output_dir = flux_training_settings['output_dir']
    output_name = flux_training_settings['output_name']
    enable_low_vram = flux_training_settings['enable_low_vram']
    blocks_to_swap = flux_training_settings['blocks_to_swap']
    use_network_weights = flux_training_settings['use_network_weights']
    network_weights_path = flux_training_settings['network_weights_path']
    generate_samples = flux_training_settings['generate_samples']
    sample_prompt_text = flux_training_settings['sample_prompt_text']
    sample_every_n_epochs = flux_training_settings['sample_every_n_epochs']
    sample_every_n_steps = flux_training_settings['sample_every_n_steps']
    num_cpu_threads_per_process = flux_training_settings['num_cpu_threads_per_process']
    num_processes = flux_training_settings['num_processes']
    attention_implementation = flux_training_settings['attention_implementation']
    optimizer_type = flux_training_settings['optimizer_type']
    max_data_loader_n_workers = flux_training_settings['max_data_loader_n_workers']
    log_type = flux_training_settings['log_type']
    log_prefix = flux_training_settings['log_prefix']
    log_dir = flux_training_settings['log_dir']
    log_tracker_name = flux_training_settings['log_tracker_name']
    offload_inactive_dit = flux_training_settings['offload_inactive_dit']
    mixed_precision = flux_training_settings['mixed_precision']
    fp8 = flux_training_settings['fp8']
    python_executable = sys.executable
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_to_disk = flux_training_settings['cache_to_disk']

    # 拼接 musubi-tuner 目录下的脚本路径
    SD_DIR = os.path.join(base_dir, 'sd-scripts')
    train_network_path = os.path.join(SD_DIR, "flux_train_network.py")
    print(python_executable, train_network_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        os.path.dirname(SD_DIR),  # LoRAMaster 根目录
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
        "--pretrained_model_name_or_path", dit_weights_path,
        "--ae",vae_path,
        "--t5xxl",t5_path,
        "--clip_l",clip_model_path,
        "--dataset_config", dataset_config,
        "--save_model_as","safetensors",
        "--mixed_precision", mixed_precision,
        "--optimizer_type", optimizer_type,
        "--learning_rate", learning_rate,
        "--gradient_checkpointing",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        "--max_data_loader_n_workers", str(max_data_loader_n_workers),
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora_flux",
        "--network_dim", str(network_dim),
        "--timestep_sampling", timestep_sampling,
        "--max_train_epochs", str(max_train_epochs),
        "--save_every_n_epochs", str(save_every_n_epochs),
        "--save_every_n_steps", str(save_every_n_steps),
        "--output_dir", output_dir,
        "--output_name", output_name,
        "--seed","42",
        "--log_with",log_type,
        "--model_prediction_type","raw"
    ]
    if cache_to_disk:
        command.extend(['--cache_latents_to_disk','--cache_text_encoder_outputs','--cache_text_encoder_outputs_to_disk'])
    if attention_implementation == 'sdpa':
        command.extend(['--sdpa'])
    elif attention_implementation == 'xformers':
        command.extend(['--xformers','--split_attn'])
    if offload_inactive_dit:
        command.extend(['--offload_inactive_dit'])
    if enable_low_vram:
        command.extend(["--blocks_to_swap", str(blocks_to_swap)])
    if use_network_weights and network_weights_path.strip():
        command.extend(["--network_weights", network_weights_path.strip()])

    if fp8:
        command.extend(['--fp8_base'])
    if generate_samples:
        prompt_file_final = make_prompt_file(
            prompt_text=sample_prompt_text
        )
        command.extend([
            "--sample_prompts", prompt_file_final,
            # "--sample_every_n_epochs", str(sample_every_n_epochs),
            "--sample_every_n_steps", str(sample_every_n_steps)
        ])

    if log_dir:
        command.extend(['--logging_dir',log_dir])
    if log_prefix:
        command.extend(['--log_prefix', log_prefix])
    if log_tracker_name:
        command.extend(['--log_tracker_name', log_tracker_name])


    def run_and_stream_output():
        global train_process
        train_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,env=env,encoding='utf-8',errors='ignore')

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
            if flux_training_settings['auto_shutdown']:
                auto_shutdown.shutdown()


    writeTrainLog("开始运行 Kontext LoRA训练命令...\n\n")
    writeTrainLog(' '.join(command))
    ui.notify("开始训练，完成前请不要离开本页面！",type='warning')
    Thread(target=run_and_stream_output).start()
def stop_train():
    msg = ''
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
    ui.label('Flux LoRA训练').classes('text-2xl font-bold')
    with ui.row().classes('w-full no-wrap gap-4'):
        with ui.column().classes('w-3/4'):
            with ui.list().props('bordered separator').classes('w-full'):
                ui.item_label('① 基本设置').props('header').classes('text-xl font-bold mb-2')
                ui.separator()
                ui.label('模型设置：').classes('font-bold mb-2').style(
                    'margin-left:10px;margin-top:10px')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('DiT Weights Path / DiT权重文件路径')
                        ui.item_label('Flux Dev模型，如flux1-dev.safetensors').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        dit_weights_path = ui.input(
                            placeholder='如I:\\train_models\\flux\\flux1-dev.safetensors',
                            value=flux_training_settings['dit_weights_path']).props(
                            'rounded outlined dense').classes('w-full')
                        bind_setting(dit_weights_path, 'dit_weights_path')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Flux AE File Path / Flux AE文件路径')
                        ui.item_label('Flux ae.sft文件的绝对路径，如:I:\\train_models\\flux\\ae.sft').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        vae_path = ui.input(placeholder='如: I:\\train_models\\flux\\ae.sft',
                                            value=flux_training_settings["vae_path"]).classes('w-full').props('rounded outlined dense')
                        bind_setting(vae_path, 'vae_path')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('CLIP_L Model Path / clip_l模型路径')
                        ui.item_label('Flux的clip_l.safetensors').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        clip_model_path = ui.input(
                            placeholder='如: I:\\train_models\\flux\\clip_l.safetensors',
                            value=flux_training_settings["clip_model_path"]).props(
                            'rounded outlined dense').classes('w-full')
                        bind_setting(clip_model_path, 'clip_model_path')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('T5 Model Path / T5模型路径')
                        ui.item_label('T5模型文件的绝对路径，如: I:\\train_models\\flux\\t5xxl_fp16.safetensors').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        t5_path = ui.input(placeholder='如：I:\\train_models\\flux\\t5xxl_fp16.safetensors',
                                           value=flux_training_settings["t5_path"]).classes('w-full').props('rounded outlined dense')
                        bind_setting(t5_path, 't5_path')

                ui.separator()
                ui.label('训练素材配置：').classes('font-bold mb-2').style(
                    'margin-left:10px;margin-top:10px')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Toml File Path / Toml文件路径')
                        ui.item_label('toml配置文件的绝对路径，如如:E:\\flux_lora_train\\config.toml').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        dataset_config = ui.input(placeholder='如:E:\\flux_lora_train\\config.toml',
                                                  value=flux_training_settings["dataset_config"]).classes(
                            'w-full').props('rounded outlined dense')
                        bind_setting(dataset_config, 'dataset_config')


            with ui.list().props('bordered separator').classes('w-full'):
                ui.item_label('② 正式训练').props('header').classes('text-xl font-bold mb-2')
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
                        max_train_epochs = ui.number(value=flux_training_settings['max_train_epochs']).props('rounded outlined dense').classes('w-1/2')
                        bind_setting(max_train_epochs, 'max_train_epochs')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Learning Rate / 学习率')
                        ui.item_label('e.g. 2e-4').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        learning_rate = ui.input(value=flux_training_settings['learning_rate']).props('rounded outlined dense').classes('w-1/2')
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
                            value=flux_training_settings['optimizer_type']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(optimizer_type, 'optimizer_type')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Network Dim / 网络维度')
                        ui.item_label('2-128，常用 4~128，不是越大越好, 低dim可以降低显存占用').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        network_dim = ui.number(value=flux_training_settings['network_dim']).props('rounded outlined dense').classes('w-1/2')
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
                            value=flux_training_settings['mixed_precision']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(mixed_precision, 'mixed_precision')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Gradient Accumulation Steps / 梯度累积步数')
                    with ui.item_section().props('side').classes('w-1/2'):
                        gradient_steps = ui.number(value=flux_training_settings['gradient_accumulation_steps']).props('rounded outlined dense').classes('w-1/2')
                        bind_setting(gradient_steps, 'gradient_accumulation_steps')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Timestep Sampling / 时间步采样')
                    with ui.item_section().props('side').classes('w-1/2'):
                        timestep_sampling = ui.input(value=flux_training_settings['timestep_sampling']).props('rounded outlined dense').classes('w-1/2')
                        bind_setting(timestep_sampling, 'timestep_sampling')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Discrete Flow Shift / 离散流移位')
                        ui.item_label('建议配置：3.15').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        discrete_shift = ui.number(value=flux_training_settings['discrete_flow_shift'], format='%.4f').props('rounded outlined dense').classes('w-1/2')
                        bind_setting(discrete_shift, 'discrete_flow_shift')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('max_data_loader_n_workers / 数据加载的最大工作线程数')
                        ui.item_label('建议2-4，如果CPU 核心多、内存大、磁盘快，可以试着调高，比如 4、8、16，能更充分利用硬件加速数据读取。').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        max_data_loader_n_workers = ui.number(value=flux_training_settings['max_data_loader_n_workers']).props('rounded outlined dense').classes('w-1/2')
                        bind_setting(max_data_loader_n_workers, 'max_data_loader_n_workers')
                ui.separator()
                ui.label('显存优化：').classes('font-bold mb-2').style('margin-left:10px;margin-top:10px')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('FP8')
                        ui.item_label('开启FP8模式，节省显存。').props('caption')
                    with ui.item_section().props('side'):

                        fp8 = ui.switch(value=flux_training_settings['fp8'])
                        bind_setting(fp8, 'fp8')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Attention Implementation / 注意力实现')
                        ui.item_label(
                            '建议使用sdpa，速度和优化程度都很好，稳定，不用额外安装库，使用xformers需要装库').props(
                            'caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        attention_implementation = ui.select(
                            ['sdpa', 'xformers'],
                            value=flux_training_settings['attention_implementation']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(attention_implementation, 'attention_implementation')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('Enable Low VRAM Mode / 启用低显存模式')
                        ui.item_label('使用低显存模式，牺牲训练速度，换取使用更低的显存').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        enable_low_vram = ui.switch(value=flux_training_settings['enable_low_vram']).props('outlined')
                        bind_setting(enable_low_vram, 'enable_low_vram')

                with ui.item().bind_visibility_from(enable_low_vram, 'value'):
                    with ui.item_section():
                        ui.item_label('Blocks to Swap / 交换块数')
                        ui.item_label(
                            '双数，最大36，数值越大，显存占用越低，训练速度越慢。5090实测，20占用13G左右，仅供参考').props(
                            'caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        blocks_to_swap = ui.number(value=flux_training_settings['blocks_to_swap']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(blocks_to_swap, 'blocks_to_swap')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('num_cpu_threads_per_process / 每个进程的CPU线程数')
                        ui.item_label('每个训练进程，开启的CPU线程数').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        num_cpu_threads_per_process = ui.number(
                            value=flux_training_settings['num_cpu_threads_per_process']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(num_cpu_threads_per_process, 'num_cpu_threads_per_process')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('num_processes / 进程数')
                        ui.item_label('训练时，开启的进程数').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        num_processes = ui.number(value=flux_training_settings['num_processes']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(num_processes, 'num_processes')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('Cache To Disk / 生成本地缓存')
                        ui.item_label('在本地生成素材的缓存文件').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        cache_to_disk = ui.switch(value=flux_training_settings['cache_to_disk']).props('outlined')
                        bind_setting(cache_to_disk, 'cache_to_disk')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('Continue Training From Existing Weights / 从已有权重继续训练')
                        ui.item_label('开启后，需要填入权重文件路径').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        use_network_weights = ui.switch(value=flux_training_settings['use_network_weights']).props(
                            'outlined')
                        bind_setting(use_network_weights, 'use_network_weights')

                with ui.item().bind_visibility_from(use_network_weights, 'value'):
                    # 权重接续训练
                    with ui.item_section():
                        ui.item_label('Weights File Path / 权重文件路径')
                        ui.item_label('开启后，需要填入权重文件路径').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        network_weights_path = ui.input(placeholder='Input weights file path / 请输入权重文件路径',
                                                        value=flux_training_settings['network_weights_path']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(network_weights_path, 'network_weights_path')
                ui.separator()
                ui.label('过程采样：').classes('font-bold mb-2').style(
                    'margin-left:10px;margin-top:10px')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Generate Samples During Training / 训练期间生成示例')
                        ui.item_label('在训练期间生成采样示例，注意，这里会拖慢训练速度！').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        generate_samples = ui.switch(value=flux_training_settings['generate_samples'])
                        bind_setting(generate_samples, 'generate_samples')


                with ui.item().bind_visibility_from(generate_samples, 'value'):
                    with ui.item_section():
                        ui.item_label('Sample Every N Steps / 每N步采样一次')
                        ui.item_label('在训练期间，每N步采一次样！').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        sample_step = ui.number(value=flux_training_settings['sample_every_n_steps']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(sample_step, 'sample_every_n_steps')
                with ui.item().bind_visibility_from(generate_samples, 'value'):
                    with ui.row().classes('w-full no-wrap gap-4'):
                        sample_prompt = ui.input('Prompt Text / 提示词',
                                                 value=flux_training_settings['sample_prompt_text']).props(
                            'outlined').classes('w-full')
                        bind_setting(sample_prompt, 'sample_prompt_text')

                ui.separator()
                ui.label('输出：').classes('font-bold mb-2').style('margin-left:10px;margin-top:10px')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Output Directory / 输出目录')
                    with ui.item_section().props('side').classes('w-1/2'):
                        output_dir = ui.input(value=flux_training_settings['output_dir']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(output_dir, 'output_dir')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('Output Name / 输出名称')
                    with ui.item_section().props('side').classes('w-1/2'):
                        output_name = ui.input(value=flux_training_settings['output_name']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(output_name, 'output_name')

                with ui.item():
                    with ui.item_section():
                        ui.item_label('Save Every N Epochs / 每N个轮次保存一次')
                        ui.item_label('每执行N轮，就保存一次Lora模型').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        save_epochs = ui.number(value=flux_training_settings['save_every_n_epochs']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(save_epochs, 'save_every_n_epochs')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Save Every N Steps / 每N步保存一次')
                        ui.item_label('每执行N步，就保存一次Lora模型').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        save_steps = ui.number(value=flux_training_settings['save_every_n_steps']).props(
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
                            value=flux_training_settings['log_type']).props(
                            'rounded outlined dense').classes('w-1/2')
                        bind_setting(log_type, 'log_type')
                ui.separator()
                ui.label('自动关机').classes('font-bold mb-2').style('margin-left:10px;margin-top:10px')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Auto Shutdown / 训练完成后自动关机')
                        ui.item_label('训练完成5分钟后，自动关机。注意：需要使用管理器启动才有效').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        aush = ui.switch(value=flux_training_settings['auto_shutdown']).props('outlined')
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


        with ui.column().classes('w-1/4').classes('bg-blue-100').style('padding:10px'):
            ui.label('参数预览').classes('text-xl font-bold mb-2')
            global settings_text
            settings_preview_label = ui.label('').bind_text_from(settings_text,'content').style("white-space: pre-wrap;")


