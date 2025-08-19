# 🦾 LoRAMaster

> **本地部署永久免费** | 云端/商业部署需书面授权  

LoRAMaster 是一个开源的 **LoRA 训练工具**，支持最新的前沿模型（如 Wan2.1 / Wan2.2 / Kontext / FramePack / Qwen-Image 等），旨在为个人用户和开发者提供高效、便捷的 LoRA 训练解决方案。

---

## 📖 功能特点

- 支持多种 LoRA 训练模式（T2V、文本到图像等）
- 完全本地部署，无需云端依赖
- 可管理训练环境和模型文件
- 集成 **TensorBoard** 查看训练状态
- 支持自定义训练参数和批量训练
- 开源，可自由修改和扩展（个人非商业用途）

---

## ✅ 支持情况

| 模型 / 模式           | 状态       |
|-------------------|------------|
| Wan 2.1 (T2V、I2V) | ✅ 支持     |
| Wan 2.2 (T2V、I2V) | ✅ 支持     |
| HunyuanVideo      | 🔧 开发中     |
| FramePack         | 🔧 开发中   |
| Kontext           | 🔧 开发中     |
| Qwen-Image        | 🔧 开发中     |


## 💻 安装与运行

1. 克隆仓库：

```bash
git clone https://github.com/AIMixer/LoRAMaster.git
cd LoRAMaster
git clone https://github.com/kohya-ss/musubi-tuner.git
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
```

3. 启动 LoRAMaster：
```bash
python main.py

```

## 📥 模型下载

| 模型名                      | 用途              | 下载链接 |
|--------------------------|-----------------|----------|
| Wan2.1 diffusion_models  | DiT权重文件路径（按需下載） | [下载](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models) |
| Wan2.2 diffusion_models                  | DiT权重文件路径（按需下載）        | [下载](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models) |
| Wan2.1 vae               | Wan VAE文件路径     | [下载](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae) |
| Wan2.1 T5                | T5模型路径          | [下载](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/blob/main/models_t5_umt5-xxl-enc-bf16.pth) |

