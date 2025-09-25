import os
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, logging
from nicegui import ui
import toml
import subprocess
from logger import logger
from datetime import datetime
import threading
import sys
import io
import tqdm
import traceback
import asyncio
import time

# ----------------- 配置 -----------------
caption_length_list = ["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261, 10)]

CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
        "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    ],
    "Rule34 tag list": [
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
    ],
    # "Danbooru tag list": [
    #     "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
    #     "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
    #     "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    # ],
    # "e621 tag list": [
    #     "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    #     "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
    #     "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    # ],
    # "Rule34 tag list": [
    #     "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
    #     "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
    #     "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
    # ],

    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

QUANTIZATION_CONFIGS = {
    "nf4": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
    },
    "int8": {
        "load_in_8bit": True,
    },
    "bf16": {},
}

EXTRA_OPTION_MAP = {
    "refer_character_name": "如果图像中有人物/角色，必须以 {name} 来称呼他们。",
    "exclude_people_info": "不要包含无法更改的人物/角色信息（如种族、性别等），但可以包含可更改的属性（如发型）。",
    "include_lighting": "包含光照信息。",
    "include_camera_angle": "包含相机角度信息。",
    "include_watermark": "包含是否有水印的信息。",
    "include_JPEG_artifacts": "包含是否存在 JPEG 压缩痕迹的信息。",
    "include_exif": "如果是照片，必须包含可能使用的相机及其参数信息，如光圈、快门速度、ISO 等。",
    "exclude_sexual": "不要包含任何性内容；保持 PG 级别。",
    "exclude_image_resolution": "不要提及图像分辨率。",
    "include_aesthetic_quality": "必须包含图像主观美学质量的信息，从低到非常高。",
    "include_composition_style": "包含图像构图风格信息，如引导线、三分法或对称性。",
    "exclude_text": "不要提及图像中的任何文字。",
    "specify_depth_field": "指定景深，并说明背景是否清晰或模糊。",
    "specify_lighting_sources": "如适用，说明可能使用的人工或自然光源。",
    "do_not_use_ambiguous_language": "不要使用任何模糊语言。",
    "include_nsfw": "包含图像是否为 SFW、暗示性或 NSFW 的信息。",
    "only_describe_most_important_elements": "只描述图像中最重要的元素。",
    "do_not_include_artist_name_or_title": "如果是艺术作品，不包含艺术家姓名或作品标题。",
    "identify_image_orientation": "识别图像方向（纵向、横向或正方形）及明显的长宽比。",
    "use_vulgar_slang_and_profanity": "使用粗俗俚语和脏话，例如（但不限于）“fucking”，“slut”，“cock”等。",
    "do_not_use_polite_euphemisms": "不要使用礼貌的委婉语——采用直接、随意的表达方式。",
    "include_character_age": "如适用，包含人物/角色的年龄信息。",
    "include_camera_shot_type": "说明图像拍摄类型，如极近景、近景、中近景、中景、牛仔镜头、中远景、远景或极远景。",
    "exclude_mood_feeling": "不要描述图像的情绪/感觉等。",
    "include_camera_vantage_height": "明确指定拍摄高度（视平线、低角度、虫眼视角、鸟瞰、无人机、屋顶等）。",
    "mention_watermark": "如果有水印，必须提及。",
    "avoid_meta_descriptive_phrases": "你的回复将用于文本生成图像模型，因此避免无用的元描述短语，例如“这张图片显示…”，“你看到的是…”，等等。",
}
predictor = None  # 不预先加载模型
settings_text = {'content': ''}


CAPTION_SETTINGS_FILE = 'dataset_manager/caption_settings.toml'
# 训练进程
caption_process = None
captionLogger = None
image_container = None
datasets = []
datasets_caption = {}

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

    toml_str = toml.dumps(caption_training_settings)
    global settings_text
    settings_text.update(content = toml_str)


caption_training_settings = load_settings()
preview_settings()

stop_flag = threading.Event()
def save_settings():
    try:
        with open(CAPTION_SETTINGS_FILE, "w", encoding="utf-8") as f:
            toml.dump(caption_training_settings, f)
    except Exception as e:
        print(f"[WARN] 保存 settings.toml 失败: {e}")
def bind_setting(ui_element, key):
    """将 UI 控件的值绑定到 caption_training_settings[key] 并自动保存"""
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

    caption_training_settings[key] = value
    save_settings()
    preview_settings()

def preview_settings():

    toml_str = toml.dumps(caption_training_settings)
    global settings_text
    settings_text.update(content = toml_str)

# ----------------- 工具函数 -----------------
def build_extra_options(selected_options: list[str], character_name: str = "Huluwa") -> list[str]:
    ret = []
    for key in selected_options:
        if key in EXTRA_OPTION_MAP:
            ret.append(EXTRA_OPTION_MAP[key].format(name=character_name))
    return ret

def build_prompt(caption_type: str, caption_length: str, extra_options: list[str] = None, user_prompt: str = "") -> str:
    print(caption_type,caption_length,extra_options,user_prompt)
    if user_prompt and user_prompt.strip():
        prompt = user_prompt.strip()
    else:
        # 选择正确的模板行
        if caption_length == "any":
            map_idx = 0
        elif isinstance(caption_length, str) and caption_length.isdigit():
            map_idx = 1  # 数字字数模板
        else:
            map_idx = 2  # 长度描述符模板

        prompt = CAPTION_TYPE_MAP[caption_type][map_idx]

    if extra_options is not None and len(extra_options) > 0:
        extra, name_input = extra_options
        prompt += " " + " ".join(extra)
    else:
        name_input = "{NAME}"
    print('prompt',prompt)
    return prompt.format(
        name=name_input,
        length=caption_length,
        word_count=caption_length,
    )

# ----------------- 模型封装 -----------------
class JoyCaptionPredictor:
    def __init__(self, quantization_mode: str = "nf4", device: str = "cuda"):
        self.device = device
        self.model_path = "fancyfeast/llama-joycaption-beta-one-hf-llava"  # 固定模型路径

        self.processor = AutoProcessor.from_pretrained(self.model_path)



        if quantization_mode == "bf16":
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            qnt_config = BitsAndBytesConfig(
                **QUANTIZATION_CONFIGS[quantization_mode],
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"]
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
                quantization_config=qnt_config
            )

        self.model.eval()

    @torch.inference_mode()
    def generate(self, image: Image.Image, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 512, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 0):
        convo = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
        convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            top_k=None if top_k == 0 else top_k,
            use_cache=True,
        )[0]
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]
        caption = self.processor.tokenizer.decode(generate_ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
        return caption.strip()


def writeTrainLog(message):
    try:
        global captionLogger
        # print('writeTrainLog', 'message:', message,'EEEnd')
        if captionLogger:
            captionLogger.push(datetime.now().strftime("%Y-%m-%d %H:%M:%S ") + message, classes='text-orange')
    except Exception as e:
        logger.info('logger error')
    logger.info(message)

def addImage(image,prompt):
    global image_container
    with image_container:
        with ui.row().classes('w-full no-wrap gap-4'):
            ui.image(image).classes('w-1/3').style('height:auto;min-height:300px;')
            ui.textarea(value=prompt).classes('w-2/3').style('height:100%;')
        ui.separator()

def save_caption_to_txt(file_path: str, content: str):

    # 写入文件（覆盖或创建）
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
def captionImage(image,convo_string,max_new_tokens,temperature,top_k,top_p):


    # Process the inputs
    inputs = predictor.processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

    # Generate the captions
    generate_ids = predictor.model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=True if float(temperature) > 0 else False,
        suppress_tokens=None,
        use_cache=True,
        temperature=float(temperature),
        top_k=float(top_k),
        top_p=float(top_p),
    )[0]

    # Trim off the prompt
    generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

    # Decode the caption
    caption = predictor.processor.tokenizer.decode(generate_ids, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)

    return caption.strip()
def run_caption():

    global datasets
    if not datasets or len(datasets) <= 0:
        ui.notify('还未加载素材，请填写素材文件夹，并点击【加载训练素材】',type='warning')
        return

    dataset_path = caption_training_settings['dataset_path']
    caption_type = caption_training_settings['caption_type']
    caption_length = caption_training_settings['caption_length']
    temperature = caption_training_settings['temperature']
    top_p = caption_training_settings['top_p']
    top_k = caption_training_settings['top_k']
    max_new_tokens = caption_training_settings['max_new_tokens']
    trigger_word = caption_training_settings['trigger_word']
    filter_word = caption_training_settings['filter_word']


    extra_options = []
    for key in EXTRA_OPTION_MAP:
        if caption_training_settings[key]:
            extra_options.append(key)


    stop_flag.clear()

    def worker():
        try:
            global predictor
            if predictor is None:
                writeTrainLog('开始下载/加载打标模型，这可能需要一些时间，请稍后...')
                predictor = JoyCaptionPredictor()
                writeTrainLog('下载/加载打标模型完成！')



            prompt = build_prompt(caption_type, caption_length, extra_options)
            writeTrainLog(prompt)
            convo = [
                {
                    "role": "system",
                    "content": 'You are a helpful image captioner.',
                },
                {
                    "role": "user",
                    "content": prompt.strip(),
                },
            ]

            # Format the conversation
            convo_string = predictor.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            assert isinstance(convo_string, str)
            global datasets,datasets_caption
            for idx, dataset in enumerate(datasets, start=1):
                if stop_flag.is_set():  # 检查是否被停止
                    writeTrainLog("[INFO] 已手动停止任务。\n")
                    break
                try:

                    # Decode the caption
                    caption = captionImage(dataset.image,convo_string,max_new_tokens,temperature,top_k,top_p)

                    writeTrainLog(f"[{idx}/{len(datasets)}] {dataset.img_filename} → {trigger_word}{caption.strip()}\n")
                    caption = trigger_word + caption.strip()
                    # 过滤标签
                    if filter_word:
                        caption = caption.replace(filter_word, "")
                    dataset.caption = caption
                    datasets_caption[dataset.img_path] = caption
                    save_caption_to_txt(dataset.caption_file_path,dataset.caption)
                except Exception as e:
                    writeTrainLog(f"[ERROR] {dataset.img_filename} 处理失败: {e}\n")
                    traceback.print_exc()
                    try:
                        # Decode the caption
                        caption = captionImage(dataset.image, convo_string, max_new_tokens, temperature, top_k,
                                               top_p)

                        writeTrainLog(f"[{idx}/{len(datasets)}] {dataset.img_filename} → {trigger_word}{caption.strip()}\n")

                        caption = trigger_word + caption.strip()
                        # 过滤标签
                        if filter_word:
                            caption = caption.replace(filter_word, "")

                        dataset.caption = caption
                        datasets_caption[dataset.img_path] = caption
                        save_caption_to_txt(dataset.caption_file_path, caption)
                    except Exception as e:
                        writeTrainLog(f"[ERROR] {dataset.img_filename} 处理失败: {e}\n")
                        traceback.print_exc()

            if not stop_flag.is_set():
                writeTrainLog("[INFO] 全部图片处理完成！\n")

        except Exception as e:
            writeTrainLog(f"[FATAL] {e}\n")
            traceback.print_exc()

    # 启动子线程
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
def run_trigger_caption():
    global datasets, datasets_caption
    if not datasets or len(datasets) <= 0:
        ui.notify('还未加载素材，请填写素材文件夹，并点击【加载训练素材】', type='warning')
        return
    trigger_word = caption_training_settings['trigger_word']
    writeTrainLog('开始只用触发词【' + trigger_word + '】打标')

    for idx, dataset in enumerate(datasets, start=1):
        dataset.caption = trigger_word
        datasets_caption[dataset.img_path] = trigger_word
        save_caption_to_txt(dataset.caption_file_path, trigger_word)
        writeTrainLog(dataset.img_filename + ' -> ' + trigger_word)
    writeTrainLog("打标完成!")

def filter_caption_word():
    global datasets, datasets_caption
    if not datasets or len(datasets) <= 0:
        ui.notify('还未加载素材，请填写素材文件夹，并点击【加载训练素材】', type='warning')
        return
    filter_word = caption_training_settings['filter_word']
    if not filter_word:
        ui.notify('还未设置过滤标签', type='warning')
        return
    writeTrainLog('开始过滤【' + filter_word + '】标签')
    for idx, dataset in enumerate(datasets, start=1):
        caption = dataset.caption
        if caption:
            caption = caption.replace(filter_word,'')
        dataset.caption = caption
        datasets_caption[dataset.img_path] = caption
        save_caption_to_txt(dataset.caption_file_path, caption)
        writeTrainLog(dataset.img_filename + ' -> ' + caption)
    writeTrainLog("打标完成!")

def stop_caption():
    if stop_flag.is_set():  # 检查是否被停止
        writeTrainLog("[INFO] 已手动停止任务。\n")
    else:
        stop_flag.set()
        writeTrainLog("[INFO] 正在尝试停止任务...\n")

class DatasetObj:
    def __init__(self, img_path,image,caption,img_filename,caption_filename,caption_filepath):
        self.img_path = img_path
        self.image = image
        self.caption = caption
        self.img_filename = img_filename
        self.caption_file_name = caption_filename
        self.caption_file_path = caption_filepath

def bind_caption(ui_element, key):
    """将 UI 控件的值绑定到 flux_training_settings[key] 并自动保存"""
    ui_element.on('update:model-value', lambda e: update_caption(key, e))

def update_caption(key ,e):
    print(key, e.args)
    """通用更新方法，支持 input / checkbox / select 等"""
    value = e.args  # 原始值
    datasets_caption[key] = value

    filename = os.path.splitext(os.path.basename(key))[0] + ".txt"
    folder_path = os.path.dirname(key)
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(value)

    writeTrainLog(key + '标签修改为：' + value)

def previewDataSets():
    global image_container
    image_container.clear()
    global datasets,datasets_caption
    with image_container:
        for dataset in datasets:
            datasets_caption[dataset.img_path] = dataset.caption
            with ui.row().classes('w-full no-wrap gap-4 items-stretch'):
                ui.image(dataset.img_path).classes('w-1/3').style('height:auto;min-height:300px;')
                with ui.column().classes('w-2/3'):
                    ui.label(dataset.img_path).classes('w-full')
                    caption_textarea = ui.textarea(value=dataset.caption).props(
                                'rounded outlined dense').classes('w-full').style('height:200px;')
                    bind_caption(caption_textarea,dataset.img_path)
                    caption_textarea.bind_value(datasets_caption,dataset.img_path)
            ui.separator()
            # time.sleep(1)
def loadDataSets():
    dataset_path = caption_training_settings['dataset_path']
    if not dataset_path:
        ui.notify('请先设置素材文件夹',type='warning')
        return

    image_files = [f for f in os.listdir(dataset_path)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]

    if not image_files:
        ui.notify("没有找到图片文件",type='warning')
        return

    ui.notify(f"共找到 {len(image_files)} 张图片")

    global datasets
    datasets = []
    for idx, img_filename in enumerate(image_files, start=1):
        img_path = os.path.join(dataset_path, img_filename)
        image = Image.open(img_path).convert("RGB")
        caption = ''
        # 读取打标文件内容

        # 获取不带扩展名的文件名
        caption_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        caption_file_path = os.path.join(dataset_path, caption_filename)
        print(caption_filename,caption_file_path)
        # 判断文件是否存在
        if os.path.exists(caption_file_path):
            with open(caption_file_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        datasets_caption[img_path] = caption
        datasets.append(DatasetObj(img_path, image, caption,img_filename,caption_filename,caption_file_path))

    previewDataSets()
    # 启动子线程
    # thread = threading.Thread(target=worker, daemon=True)
    # thread.start()
# ----------------- NiceGUI 前端 -----------------
def draw_ui():

    ui.label('LoRA训练自动打标（可根据个人需求，灵活配置，生成定制化标签）').classes('text-2xl font-bold')
    with ui.row().classes('w-full no-wrap gap-4'):
        with ui.column().classes('w-2/3'):
            with ui.list().props('bordered separator').classes('w-full'):
                ui.item_label('① 基本设置').props('header').classes('text-xl font-bold mb-2')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('素材文件夹')
                        ui.item_label('素材文件夹的绝对路径，如:E:\\train\\aiblender_v2\\images').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        with ui.row().classes('w-full no-wrap gap-4'):
                            dataset_path = ui.input(placeholder='如: E:\\train\\aiblender_v2\\images',
                                                value=caption_training_settings["dataset_path"]).classes('w-2/3').props(
                                'rounded outlined dense')
                            bind_setting(dataset_path, 'dataset_path')
                            ui.button('加载训练素材', icon='history', on_click=loadDataSets).classes('w-1/3')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('触发词')
                        ui.item_label('将会拼接到标签的最前面，如aijbs,').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        trigger_word = ui.input(placeholder='如: aijbs,',
                                            value=caption_training_settings["trigger_word"]).classes('w-full').props(
                            'rounded outlined dense')
                        bind_setting(trigger_word, 'trigger_word')


            with ui.list().props('bordered separator').classes('w-full'):
                ui.item_label('② 打标设置').props('header').classes('text-xl font-bold mb-2')
                ui.separator()
                with ui.item():
                    with ui.item_section():
                        ui.item_label('打标类型')
                        ui.item_label('根据打标类型的不同，生成不同风格的标签').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        caption_type = ui.select(
                            list(CAPTION_TYPE_MAP.keys()),
                            value=caption_training_settings['caption_type']).props(
                            'rounded outlined dense').classes(
                            'w-full')
                        bind_setting(caption_type, 'caption_type')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('打标长度')
                        ui.item_label('根据打标长度的设置，生成不同长度的标签').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        caption_length = ui.select(caption_length_list,
                            value=caption_training_settings['caption_length']).props(
                            'rounded outlined dense').classes(
                            'w-full')
                        bind_setting(caption_length, 'caption_length')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Temperature / 温度')
                        ui.item_label('数值越高越随机，越低越稳定').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        temperature = ui.slider(min=0, max=2, step=0.01, value=caption_training_settings['temperature']).props('label-always')
                        bind_setting(temperature, 'temperature')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Top-p / 核采样概率')
                        ui.item_label('数值低 → 模型只选最可能的词，结果安全但单一;数值高 → 模型可以选更多可能的词，结果更自由、创意更多').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        top_p = ui.slider(min=0, max=1, step=0.01,
                                                value=caption_training_settings['top_p']).props('label-always')
                        bind_setting(top_p, 'top_p')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('Max New Tokens / 最大新生成词数')
                        ui.item_label('模型生成的最大 token（单词或子词）数量，达到上限 → 生成自动停止，防止内容过长').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        max_new_tokens = ui.slider(min=0, max=2048, step=1,
                                                value=caption_training_settings['max_new_tokens']).props('label-always')
                        bind_setting(max_new_tokens, 'max_new_tokens')
                with ui.item():
                    with ui.item_section():
                        ui.item_label('过滤词')
                        ui.item_label('打标时去掉不想要的标签，如:CGI').props('caption')
                    with ui.item_section().props('side').classes('w-1/2'):
                        with ui.row().classes('w-full no-wrap gap-4'):
                            filter_word = ui.input(placeholder='如: CGI',
                                                value=caption_training_settings["filter_word"]).classes('w-full').props(
                                'rounded outlined dense')
                            bind_setting(filter_word, 'filter_word')
                with ui.expansion('附加参数设置', icon='work').classes('w-full'):
                    with ui.list().props('bordered separator').classes('w-full'):
                        for key, label_text in EXTRA_OPTION_MAP.items():
                            with ui.item():
                                with ui.item_section():
                                    ui.item_label(key)
                                    ui.item_label(label_text).props('caption')
                                with ui.item_section().props('side'):
                                    fp8 = ui.switch(value=caption_training_settings[key])
                                    bind_setting(fp8, 'fp8')

                with ui.item():
                    with ui.row().classes('w-full no-wrap gap-4'):
                        ui.button('只用触发词打标', on_click=run_trigger_caption).classes('w-1/2')
                        ui.button('过滤标签', on_click=filter_caption_word).classes('w-1/2')
                with ui.item():
                    with ui.row().classes('w-full no-wrap gap-4'):
                        ui.button('Start Captioning / 开始打标',color='green', on_click=run_caption).classes('w-1/2')
                        ui.button('Stop Captioning / 停止打标', color='red', on_click=stop_caption).classes('w-1/2')

                with ui.item():
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label('输出日志').classes('text-xl font-bold')
                        ui.button('清空日志', on_click=lambda: captionLogger.clear())
                with ui.row().classes('w-full'):
                    global captionLogger
                    captionLogger = ui.log().classes('w-full h-30').style('height:500px')

        with ui.column().classes('w-1/3').style('padding:10px'):
            ui.label('素材预览').classes('text-xl font-bold mb-2')
            global image_container
            image_container = ui.row().classes('w-full')

# ----------------- 运行 -----------------
if __name__ == "__main__":
    draw_ui()
    ui.run()