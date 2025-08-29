from nicegui import ui

active_menu = "dashboard"

# 左侧菜单项
menu_items = [
    {"label": "关于LoRA训练大师", "icon": "rocket", "key": "/"},
    {"label": "万相视频 LoRA训练", "icon": "rocket", "key": "/Wan"},
    {"label": "Flux Kontext LoRA训练", "icon": "settings", "key": "/FluxKontext"},
    {"label": "Qwen Image LoRA训练", "icon": "rocket", "key": "/QwenImage"},
    # {"label": "混元视频 LoRA训练", "icon": "rocket", "key": "/HunyuanVideo"},
    # {"label": "FramePack LoRA训练", "icon": "download", "key": "/FramePack"},
    {"label": "TensorBoard 仪表盘", "icon": "rocket", "key": "/Tensorboard"},
]

def with_layout(content_func):
    """
    包装页面内容，添加顶部栏、侧边栏等公共布局
    content_func: 页面具体内容定义的函数
    """
    # 顶部菜单栏
    ui.add_head_html("""
        <style>
            * {
                text-transform: none !important;
            }
        </style>
    """)
    with ui.header().classes('items-center justify-between bg-primary text-white'):
        with ui.row().classes('items-center'):
            ui.image('assets/logo.png').style('width:50px;height:50px')
            ui.label('LoRAMaster / LoRA训练大师 by AI搅拌手').classes('text-xl font-bold ml-2')
        with ui.row().classes('items-center'):
            ui.label('QQ交流群：551482703')
            ui.html("""
                <a href="https://space.bilibili.com/1997403556" target="_blank">联系作者：AI搅拌手</a>
                <a href="https://github.com/AIMixer/LoRAMaster" target="_blank">Github</a>
            """)
        # ui.button(icon='menu', on_click=lambda: right_drawer.toggle()).props('flat color=white')


    # 左侧导航栏
    with ui.left_drawer().classes('bg-blue-50'):
        for item in menu_items:
            is_active = (item["key"] == active_menu)
            button = ui.button(on_click=lambda key=item["key"]: switch_page(key)).props('flat').classes(
                f'flex items-center w-full text-left p-3 rounded-lg transition-colors '
                + ('bg-gray-800 border-l-4 border-blue-500' if is_active else 'hover:bg-gray-800')
            )
            with button:
                # ui.icon(item["icon"]).classes('mr-3')
                ui.label(item["label"]).classes('text-base')
        # 显示广告内容
        with ui.element().classes('fixed bottom-5 left-5 w-[260px]'):
            with ui.link(target='https://space.bilibili.com/1997403556', new_tab=True):
                ui.image('assets/follow.png').classes('w-full h-[100px]').style('margin-bottom:5px')
            with ui.link(target='https://comfyit.cn', new_tab=True):
                ui.image('assets/master.png').classes('w-full h-[100px]')
    # 右侧抽屉栏
    # with ui.right_drawer(fixed=False).classes('bg-blue-100').props('bordered') as right_drawer:
    #     ui.label('📌 右侧抽屉内容')

    # 页面主体内容（传入的函数）
    with ui.column().classes('w-full').style('width:100%;height:100%'):
        content_func()

    # 底部栏
    with ui.footer().style('background-color: #3874c8'):
        ui.label('版权声明：本软件由AI搅拌手开发，在个人与单位的本地部署环境下，永久免费使用，不收取任何费用。任何云端服务商、SaaS 平台或通过互联网向第三方提供本软件运行环境的行为，均须事先获得作者的授权。未经授权的云端部署、商业化运营或提供在线访问的行为，将依法追究法律责任。').classes('text-white ml-2')

    def switch_page(key: str):
        global active_menu
        active_menu = key
        # ui.navigate.to(key, new_tab=False)
        ui.run_javascript('window.location.href="' + key + '";')
        # ui.run_javascript("location.reload()")  # 简化：刷新后重新渲染 active
        # 实际中可以改为内容区域 update 替代 reload


