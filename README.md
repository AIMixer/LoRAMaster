# 🦾 LoRAMaster

> **本地部署永久免费** | 云端/商业部署需书面授权  

LoRAMaster 是一个开源的 **LoRA 训练工具**，支持最新的前沿模型（如 Wan2.1 / Wan2.2 / Kontext / Hunyuan / Qwen-Image / Qwen-Image Edit 等），旨在为个人用户和开发者提供高效、便捷的 LoRA 训练解决方案。

---

## 📖 功能特点

- 支持多种 LoRA 训练模式（Wan2.1 / Wan2.2 / Kontext / Hunyuan / Qwen-Image / Qwen-Image Edit 等）
- 支持本地部署和云端部署
- 集成 **TensorBoard** 查看训练状态
- 支持自定义训练参数
- 本地使用，完全开源免费
---

## ✅ 支持情况

| 模型 / 模式           | 状态       |
|-------------------|------------|
| Wan 2.1 (T2V、I2V) | ✅ 支持     |
| Wan 2.2 (T2V、I2V) | ✅ 支持     |
| Kontext           | ✅ 支持     |
| Qwen-Image        | ✅ 支持      |
| Qwen-Image Edit   | ✅ 支持      |
| HunyuanVideo      | ✅ 支持     |
| Flux              | ✅ 支持     |


## 视频教程
### 本地训练教程
1. [LoRA训练大师简介与安装方法](https://www.bilibili.com/video/BV1kdeuzvE2j/)
2. [Wan2.1 文生视频LoRA训练教程（以人物角色为例）](https://www.bilibili.com/video/BV19BYUz4EHz)
3. [Wan2.1 图生视频LoRA训练教程（以视频特效为例）](https://www.bilibili.com/video/BV1sAeqz1ETM)
4. [Wan2.2 文生视频LoRA训练教程（以人物角色为例）](https://www.bilibili.com/video/BV1N6exzDEZK)
5. [Wan2.2 图生视频LoRA训练教程（以视频特效为例）](https://www.bilibili.com/video/BV1JkekzWEzn)
6. [Kontext LoRA训练教程（以提取花纹LoRA为例）](https://www.bilibili.com/video/BV1Pve9zZENV)
7. [Qwen Image训练教程（以人物角色LoRA为例）](https://www.bilibili.com/video/BV1sPhXzJEJx)
8. [Qwen Image Edit训练教程（以提取花纹LoRA为例）](https://www.bilibili.com/video/BV1mKhezNEPz)
9. [Flux 训练教程（以人物角色LoRA为例）](https://www.bilibili.com/video/BV1utHezUEB9)
10. [Flux 风格LoRA训练教程](https://www.bilibili.com/video/BV1D1pTz5EXG)
11. 持续更新中...
### 云端训练教程（仙宫云）
1. [Wan2.1 文生视频LoRA训练教程（以人物角色为例）](https://www.bilibili.com/video/BV16WagzbEog)
2. [Wan2.1 图生视频LoRA训练教程（以视频特效为例）](https://www.bilibili.com/video/BV1tHatz9Ej7)
3. [Wan2.2 文生视频LoRA训练教程（以人物角色为例）](https://www.bilibili.com/video/BV1dCaqz8EpN)
4. [Wan2.2 图生视频LoRA训练教程（以视频特效为例）](https://www.bilibili.com/video/BV163aizYEWb)
5. [Kontext LoRA训练教程（以提取花纹LoRA为例）](https://www.bilibili.com/video/BV1HhaqzzEdR)
6. [Qwen Image训练教程（以人物角色LoRA为例）](https://www.bilibili.com/video/BV1f8YTzDEtf)
7. [Qwen Image Edit训练教程（以提取花纹LoRA为例）](https://www.bilibili.com/video/BV1AnYTzZEUG)
8. [Flux 训练教程（以人物角色LoRA为例）](https://www.bilibili.com/video/BV16THezwEwc)
9. 持续更新中...

## 💻 本地安装与运行

### 方式一：本地一键整合包
👉 [点击这里，下载一键整合包](https://comfyit.cn/article/401)

### 方式二：本地手动安装
Python要求：3.12（作者基于3.12.10测试）

1. 克隆仓库：

```bash
git clone https://github.com/AIMixer/LoRAMaster.git
cd LoRAMaster
git clone https://github.com/kohya-ss/musubi-tuner.git
git clone -b sd3 --single-branch https://github.com/kohya-ss/sd-scripts
```

2. 创建虚拟环境并安装依赖：
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd musubi-tuner
pip install -e .
cd ..
pip install -r requirements.txt
cd sd-scripts
pip install -r requirements.txt
cd ..
```

3. 启动 LoRAMaster：
```bash
python main.py
```

## 单独启动方式
1. 进入LoRAMaster根目录
2. 打开CMD，依次运行以下命令
```bash
.venv\Scripts\activate
python main.py
```

## 📥 模型下载

| 模型名                      | 用途              | 下载链接 |
|--------------------------|-----------------|----------|
| Wan2.1 diffusion_models  | DiT权重文件路径（按需下載） | [下载](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models) |
| Wan2.2 diffusion_models             | DiT权重文件路径（按需下載） | [下载](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models) |
| Wan2.1 vae               | Wan VAE文件路径     | [下载](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae) |
| Wan2.1 T5                | T5模型路径          | [下载](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/blob/main/models_t5_umt5-xxl-enc-bf16.pth) |
| CLIP Model                | Clip模型          | [下载](https://www.modelscope.cn/models/muse/open-clip-xlm-roberta-large-vit-huge-14/files) |


## ⚖️ 使用条款

1. **本地使用完全免费**  
   - 本工具仅供本地环境运行。  
   - 不允许在未经作者本人授权，直接或间接在任何 **云端平台、在线服务或远程服务器** 中部署。  

2. **禁止修改和二次分发**  
   - 未经作者书面许可，不得修改、拆分、二次开发或再分发本项目。  
   - 不得将本项目代码或衍生品用于 **商业化分发** 或 **SaaS 服务**。  

3. **云端使用需授权**  
   - 若需将本项目部署到云端（如私有服务器、商用云服务、远程训练平台等），必须事先获得作者明确授权。  

4. **个人学习与研究**  
   - 欢迎个人学习、研究、实验使用，但请严格遵守以上限制。  

---

## 🔑 授权方式
如需 **商业合作、云端授权或定制功能**，请联系作者：  
- B站主页：[AI搅拌手](https://space.bilibili.com/1997403556)
- QQ交流群：551482703
- 作者QQ：3697688140