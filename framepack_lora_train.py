from nicegui import ui

def draw_ui():
    ui.label('FramePack Lora训练正在开发中，敬请期待...').classes('text-xl font-bold')
    with ui.row().classes('w-full no-wrap gap-4'):
        with ui.link(target='https://space.bilibili.com/1997403556',new_tab=True):
            ui.label('关注作者（AI搅拌手），获取最新动态').classes('text-xl font-bold')