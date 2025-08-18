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

## 💻 安装与运行

1. 克隆仓库：

```bash
git clone https://github.com/AIMixer/LoRAMaster.git
cd LoRAMaster
```

2. 创建虚拟环境并安装依赖：
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
cd musubi-tuner
pip install -e .
cd ..
```

3. 启动 LoRAMaster：
```bash
python main.py

```