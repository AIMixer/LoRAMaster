from nicegui import ui

def draw_ui():
    # 顶部主标题
    ui.markdown("""
    # 🦾 LoraMaster - LoRA训练大师
    > **本地部署永久免费** | 云端部署需授权  
    > 支持前沿模型（如 Wan2.1 / Wan2.2 / Kontext / FramePack / Qwen-Image 等）的 LoRA 训练工具
    """).classes("m-4")

    # 软件说明
    ui.markdown("""
    ### 📖 软件说明
    LoraMaster 是专为 **LoRA 训练** 设计的开源工具，支持最新的文生图、文生视频、图生视频等模型。  
    集成了 **训练参数配置、训练进度可视化、一键开启训练** 等实用功能。

    **主要特点：**
    
    - 🎯 支持 Wan 系列、Kontext、Qwen_Image、混元、FramePack 等前沿模型
    - 📊 内置 TensorBoard 仪表盘，实时查看训练曲线
    - ⚡ 预缓存 & 正式训练分离，提高训练效率
    - 🗂 低显存模式，让本地消费级显卡可以训练Lora

    **使用范围：**
    
    - ✅ **个人与单位本地部署**：永久免费  
    - ⚠️ **云端/在线服务商**：必须获得作者书面授权
    
    """).classes("m-4")

    # 作者信息
    ui.markdown("### 👤 作者信息").classes("m-4")
    with ui.column().classes('m-4'):
        ui.html("""
            <ul>
                <li>作者：<b>AI搅拌手</b></li><br />
                <li><a href="https://space.bilibili.com/1997403556" target="_blank" style="text-decoration: underline;">B站首页：AI搅拌手</a></li><br />
                <li><a href="https://loramaster.com" target="_blank" style="text-decoration: underline;">Lora训练大师官网：https://loramaster.com</a></li><br />
                <li><a href="https://comfyit.cn" target="_blank" style="text-decoration: underline;">ComfyUI搅拌站：https://comfyit.cn</a></li><br />
                <li><a href="https://comfyit.cn/article/286" target="_blank" style="text-decoration: underline;">ComfyUI管理大师：https://comfyit.cn/article/286</a></li><br />
                <li>LoRA训练交流群：559826331；ComfyUI技术交流群：551482703</li>
            </ul>
        """)

    # 捐助支持
    ui.markdown("""
    ### 💝 捐助支持
    如果你觉得本软件有帮助，可以联系并捐赠作者，捐助者将出现在本页面以及官网的捐赠者名单里
    """).classes("m-4")

    # 版权声明
    ui.markdown("""
    ### 📜 版权声明
    本软件 **在个人与单位的本地部署环境下，永久免费使用**。  
    任何云端部署、SaaS 化运营或通过互联网向第三方提供本软件运行环境的行为，必须获得作者授权。  
    违者将依法追究法律责任。
    """).classes("m-4")