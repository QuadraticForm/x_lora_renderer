bl_info = {
    "name": "x_lora_renderer",
    "author": "xuxing",
    "blender": (4, 0, 0),
    "version": (0, 0, 1),
    "location": "View3D > Sidebar",
    "description": "",
    "warning": "",
    "category": "Render"
}

import bpy
import sys
from . import auto_load
# from . import properties

auto_load.init()

def register():
    auto_load.register()

    #bpy.types.Scene.x_anim = bpy.props.PointerProperty(type=properties.x_anim_properties)


def unregister():
    auto_load.unregister()

    auto_load.cleanse_modules(__name__)

    #del bpy.types.Scene.x_anim


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
    register()
