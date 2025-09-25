import os
from nicegui import ui
import toml
from logger import logger
from datetime import datetime
import threading
import shutil
from PIL import Image, UnidentifiedImageError, features

settings_text = {'content': ''}

CAPTION_SETTINGS_FILE = 'dataset_manager/image_convert_settings.toml'

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
def convertSuffix():
    dataset_path = preprocess_training_settings["dataset_path"]
    target_suffix = preprocess_training_settings["target_suffix"]

    if not dataset_path:
        ui.notify("请填写素材目录",type="warning")
        return
    if not target_suffix:
        ui.notify("请填写目标格式",type="warning")
        return

    target_suffix = target_suffix.replace(".", "")
    input_folder = os.path.abspath(dataset_path)
    backup_folder = os.path.join(input_folder, "loramaster_backup")
    abs_backup = os.path.abspath(backup_folder)
    def worker():

        """
           - 先把 input_folder 完整备份到 input_folder/loramaster_backup
           - 再批量读取 input_folder 下的图片，统一转成 JPG 格式保存
           """
        # 1. 设置备份目录
        backup_folder = os.path.join(dataset_path, "loramaster_backup")

        # 2. 备份
        if os.path.exists(backup_folder):
            writeTrainLog(f"⚠️ 备份目录已存在：{backup_folder}，跳过备份。")
        else:
            writeTrainLog("开始备份...")
            shutil.copytree(dataset_path, backup_folder)
            writeTrainLog(f"✅ 已完成备份：{backup_folder}")

        # 3. 支持的图片格式
        supported_exts = {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".webp"}

        converted = 0
        scanned = 0
        failures = []
        overwrite = True
        quality = 95
        remove_original = True
        for root, dirs, files in os.walk(dataset_path):
            root_abs = os.path.abspath(root)
            # 跳过备份目录
            if root_abs == abs_backup or root_abs.startswith(abs_backup + os.sep):
                continue

            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in supported_exts:
                    continue

                scanned += 1
                input_path = os.path.join(root, fname)
                output_path = os.path.join(root, os.path.splitext(fname)[0] + '.' + target_suffix)

                # 如果不是 overwrite 且目标已存在，则跳过
                if not overwrite and os.path.exists(output_path):
                    writeTrainLog(f"跳过（已存在）: {output_path}")
                    continue

                try:
                    with Image.open(input_path) as img:
                        # 如果是动图（webp/gif），取第一帧
                        if getattr(img, "is_animated", False):
                            try:
                                img.seek(0)
                            except Exception:
                                pass

                        rgb = img.convert("RGB")
                        target_format = target_suffix.upper()
                        if target_suffix.upper() == 'JPG':
                            target_format = "JPEG"
                        rgb.save(output_path, target_format)
                    converted += 1
                    writeTrainLog(f"✅ 转换: {input_path} -> {output_path}")

                    # 如果用户要求删除原文件（备份已存在）
                    if remove_original:
                        try:
                            # 防止意外删除刚保存的 JPG（名字相同但扩展不同一般无此问题）
                            if os.path.exists(output_path) and os.path.abspath(input_path) != os.path.abspath(output_path):
                                os.remove(input_path)
                                writeTrainLog(f"🗑 已删除原文件: {input_path}")
                        except Exception as e:
                            writeTrainLog(f"⚠️ 删除原文件失败: {input_path}, 错误: {e}")

                except UnidentifiedImageError:
                    failures.append((input_path, "UnidentifiedImageError"))
                    writeTrainLog(f"❌ 无法识别为图片: {input_path}")
                except Exception as e:
                    failures.append((input_path, str(e)))
                    writeTrainLog(f"❌ 转换失败: {input_path}, 错误: {e}")

        # 4. 总结
        writeTrainLog("----- 完成 -----")
        writeTrainLog(f"扫描到文件: {scanned}")
        writeTrainLog(f"成功转换: {converted}")
        writeTrainLog(f"失败数量: {len(failures)}")
        if failures:
            writeTrainLog("失败样例（最多 20 条）:")
            for p, err in failures[:20]:
                writeTrainLog(f" - {p}  => {err}")
        writeTrainLog("转换完成，共转换【" + str(converted) + "】张图片")
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
# ----------------- NiceGUI 前端 -----------------
def draw_ui():

    ui.label('图片素材转格式').classes('text-2xl font-bold')
    with ui.row().classes('w-full no-wrap gap-4'):
        with ui.column().classes('w-3/4'):
            with ui.list().props('bordered separator').classes('w-full'):
                ui.item_label('基本设置').props('header').classes('text-xl font-bold mb-2')
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
                ui.separator()
                ui.item_label('批量转格式').props('header').classes('text-xl font-bold mb-2')
                ui.separator()

                with ui.item():
                    with ui.item_section():
                        ui.item_label('目标格式')
                        ui.item_label('将".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".webp"图片，转化成为指定格式，如JPG').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        target_suffix = ui.input(placeholder='如: aijbs,',
                                            value=preprocess_training_settings["target_suffix"]).classes('w-full').props(
                            'rounded outlined dense')
                        bind_setting(target_suffix, 'target_suffix')
                with ui.item():
                    with ui.row().classes('w-full no-wrap gap-4'):
                        ui.button('开始转换',color='green', on_click=convertSuffix).classes('w-full')



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