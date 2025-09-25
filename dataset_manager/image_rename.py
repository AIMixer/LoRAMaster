import os
from nicegui import ui
import toml
from logger import logger
from datetime import datetime
import threading
import shutil
from PIL import Image, UnidentifiedImageError, features
from glob import glob

settings_text = {'content': ''}

CAPTION_SETTINGS_FILE = 'dataset_manager/image_rename_settings.toml'

captionLogger = None

def load_settings() -> dict:
    if os.path.exists(CAPTION_SETTINGS_FILE):
        try:
            with open(CAPTION_SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = toml.load(f)
                return settings
        except Exception:
            return {}
    else:
        print("打标配置文件不存在")
        return {}
def preview_settings():

    toml_str = toml.dumps(preprocess_training_settings)
    global settings_text
    settings_text.update(content = toml_str)


preprocess_training_settings = load_settings()
preview_settings()

def save_settings():
    try:
        with open(CAPTION_SETTINGS_FILE, "w", encoding="utf-8") as f:
            toml.dump(preprocess_training_settings, f)
    except Exception as e:
        print(f"[WARN] 保存 settings.toml 失败: {e}")
def bind_setting(ui_element, key):
    """将 UI 控件的值绑定到 preprocess_training_settings[key] 并自动保存"""
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

    preprocess_training_settings[key] = value
    save_settings()
    preview_settings()

def preview_settings():

    toml_str = toml.dumps(preprocess_training_settings)
    global settings_text
    settings_text.update(content = toml_str)


def writeTrainLog(message):
    try:
        global captionLogger
        # print('writeTrainLog', 'message:', message,'EEEnd')
        if captionLogger:
            captionLogger.push(datetime.now().strftime("%Y-%m-%d %H:%M:%S ") + message, classes='text-orange')
    except Exception as e:
        logger.info('logger error')
    logger.info(message)

    # 启动子线程
    # thread = threading.Thread(target=worker, daemon=True)
    # thread.start()
def run_rename():
    dataset_path = preprocess_training_settings["dataset_path"]
    target_suffix = preprocess_training_settings["target_suffix"]
    target_num = preprocess_training_settings["target_num"]
    target_prefix = preprocess_training_settings["target_prefix"]

    if not dataset_path:
        ui.notify("请填写素材目录",type="warning")
        return
    if not target_suffix:
        ui.notify("请填写目标格式",type="warning")
        return
    if not target_num:
        ui.notify("请填写命名位数",type="warning")
        return

    target_suffix = target_suffix.replace(".", "")
    def worker():
        updated_count = 0

        files = glob(os.path.join(dataset_path, f"*{target_suffix}"))
        files.sort()  # Ensure files are sorted alphabetically

        if not files:
            writeTrainLog("未找到需要重命名的文件")
            return


        for index, file_path in enumerate(files, start=1):
            directory, old_name = os.path.split(file_path)
            new_name = f"{target_prefix}{str(index).zfill(int(target_num))}.{target_suffix}"
            new_path = os.path.join(directory, new_name)

            os.rename(file_path, new_path)

            writeTrainLog(f"文件重命名: {file_path} -> {new_path}")

            updated_count += 1
        writeTrainLog(f"重命名完成，共重命名了{updated_count}个文件")
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
# ----------------- NiceGUI 前端 -----------------
def draw_ui():

    ui.label('素材重命名').classes('text-2xl font-bold')
    with ui.row().classes('w-full no-wrap gap-4'):
        with ui.column().classes('w-3/4'):
            with ui.list().props('bordered separator').classes('w-full'):
                ui.item_label('重命名设置').props('header').classes('text-xl font-bold mb-2')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('素材文件夹')
                        ui.item_label('素材文件夹的绝对路径，如:E:\\train\\aiblender_v2\\images').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        with ui.row().classes('w-full no-wrap gap-4'):
                            dataset_path = ui.input(placeholder='如: E:\\train\\aiblender_v2\\images',
                                                value=preprocess_training_settings["dataset_path"]).classes('w-full').props(
                                'rounded outlined dense')
                            bind_setting(dataset_path, 'dataset_path')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('素材格式')
                        ui.item_label('指定需要重命名的素材格式，如jpg').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        with ui.row().classes('w-full no-wrap gap-4'):
                            target_suffix = ui.input(placeholder='如: jpg',
                                                value=preprocess_training_settings["target_suffix"]).classes('w-full').props(
                                'rounded outlined dense')
                            bind_setting(target_suffix, 'target_suffix')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('命名位数')
                        ui.item_label('如4，则命名为0001.jpg,0002.jpg').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        with ui.row().classes('w-full no-wrap gap-4'):
                            target_num = ui.number(placeholder='如: 4',
                                                value=preprocess_training_settings["target_num"]).classes('w-full').props(
                                'rounded outlined dense')
                            bind_setting(target_num, 'target_num')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('命名前缀')
                        ui.item_label('如需加前缀，则填写命名后的前缀，如AIJBS，命名后为AIJBS0001').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        with ui.row().classes('w-full no-wrap gap-4'):
                            target_prefix = ui.input(placeholder='如: AIJBS',
                                                     value=preprocess_training_settings["target_prefix"]).classes(
                                'w-full').props(
                                'rounded outlined dense')
                            bind_setting(target_prefix, 'target_prefix')

                with ui.item():
                    with ui.row().classes('w-full no-wrap gap-4'):
                        ui.button('开始重命名',color='green', on_click=run_rename).classes('w-full')



                with ui.item():
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label('输出日志').classes('text-xl font-bold')
                        ui.button('清空日志', on_click=lambda: captionLogger.clear())
                with ui.row().classes('w-full'):
                    global captionLogger
                    captionLogger = ui.log().classes('w-full h-30').style('height:500px')

        with ui.column().classes('w-1/4').style('padding:10px'):
            pass

# ----------------- 运行 -----------------
if __name__ == "__main__":
    draw_ui()
    ui.run()