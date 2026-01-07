import bpy
import os
import re
from bpy.types import Operator, Panel
from bpy.props import StringProperty

# -------------------------
# Helpers
# -------------------------

_WINDOWS_FORBIDDEN = r'[<>:"/\\|?*\x00-\x1F]'

def safe_open_file(filepath: str, mode: str, encoding: str = 'utf-8'):
    """Open a file, creating its parent directories if they don't exist."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    return open(filepath, mode, encoding=encoding)

def sanitize_filename(name: str) -> str:
    """Make filename safe-ish for Windows/macOS/Linux."""
    name = re.sub(_WINDOWS_FORBIDDEN, "_", name)
    name = name.strip().strip(".")
    return name or "untitled"

def resolve_output_dir(p:str) -> str:
    # fallback: blend directory
    base = bpy.path.abspath("//")

    if not p:
        return base

    # normalize slashes
    p = p.replace("\\", "/")

    # Blender relative path
    if p.startswith("//"):
        return bpy.path.abspath(p)

    # Detect absolute paths (Windows + Unix)
    is_abs_win = len(p) >= 2 and p[1] == ":"  # "C:/..."
    is_abs_unc = p.startswith("//")           # "//server/share"
    is_abs_unix = p.startswith("/")           # "/tmp/..."
    if is_abs_win or is_abs_unc or is_abs_unix:
        return os.path.abspath(p)

    # Otherwise treat as relative to blend file directory
    return os.path.abspath(os.path.join(base, p))


def get_marker_name(scene: bpy.types.Scene, frame: int) -> str:
    for m in scene.timeline_markers:
        if m.frame == frame:
            return m.name
    return ""

def make_format_context(scene: bpy.types.Scene, camera: bpy.types.Object) -> dict:
    frame = scene.frame_current
    tag = camera.get("tag", "") if camera else ""
    marker = get_marker_name(scene, frame)
    cam_name = camera.name if camera else "Camera"
    return {
        "cam": cam_name,
        "frame": frame,
        "tag": tag,
        "marker": marker,
    }

def safe_format(template: str, ctx: dict) -> str:
    try:
        return template.format(**ctx)
    except Exception as e:
        return f"[FORMAT ERROR: {e}]"

def clean_caption(s: str) -> str:
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]  # drop empty tokens
    return ", ".join(parts)

def safe_format_caption(template: str, ctx: dict) -> str:
    return clean_caption(safe_format(template, ctx))

def pick_preview_camera(context) -> bpy.types.Object | None:
    """Prefer selected camera, else scene camera, else first camera in scene."""
    scene = context.scene

    obj = context.object
    if obj and obj.type == "CAMERA":
        return obj

    if scene.camera and scene.camera.type == "CAMERA":
        return scene.camera

    cams = [o for o in scene.objects if o.type == "CAMERA"]
    return cams[0] if cams else None

def collect_visible_cameras(scene: bpy.types.Scene, view_layer: bpy.types.ViewLayer) -> list[bpy.types.Object]:
    """Collect cameras that are visible in the given view layer."""
    return [
        obj for obj in scene.objects
        if obj.type == 'CAMERA'
        and obj.name in view_layer.objects              # Not excluded from current View Layer
        and obj.visible_get(view_layer=view_layer)      # Visible in current View Layer
        and not obj.hide_render                         # (Optional) Not disabled for rendering
    ]

def collect_current_visible_cameras() -> list[bpy.types.Object]:
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    return collect_visible_cameras(scene, view_layer)

# -------------------------
# Operators
# -------------------------

class XLR_OT_GenerateCaptions(Operator):
    bl_idname = "xlr.generate_captions"
    bl_label = "Generate Captions (.txt)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        out_dir = resolve_output_dir(scene.render.filepath)

        cams = collect_visible_cameras(scene, context.view_layer)

        if not cams:
            self.report({'WARNING'}, "No cameras found")
            return {'CANCELLED'}

        orig_frame = scene.frame_current

        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)
            for cam in cams:
                ctx = make_format_context(scene, cam)

                raw_name = safe_format(scene.xlr_filename_template.strip(), ctx)
                filename = sanitize_filename(raw_name)

                tagpath = os.path.join(out_dir, f"{filename}.txt")
                caption = safe_format_caption(scene.xlr_caption_template, ctx)

                with safe_open_file(tagpath, 'w', encoding='utf-8') as f:
                    f.write(caption)

        scene.frame_set(orig_frame)
        self.report({'INFO'}, f"Captions generated to: {out_dir}")
        return {'FINISHED'}


class XLR_OT_RenderAllCameras(Operator):
    bl_idname = "xlr.render_all_cameras"
    bl_label = "Render All Cameras"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None

    def invoke(self, context, event):
        # init state (do NOT override __init__ in Blender operators)
        self.current_camera_index = 0
        self.cameras = []
        self.original_camera = None
        self.original_render_filepath = ""
        self._timer = None

        self.scene = context.scene
        self.view_layer = context.view_layer

        scene = context.scene
        view_layer = context.view_layer  # 或 bpy.context.view_layer

        self.original_camera = scene.camera
        self.original_render_filepath = scene.render.filepath

        # 只收集当前 view layer 里“可见/生效”的相机
        self.cameras = collect_visible_cameras(scene, view_layer)

        if not self.cameras:
            self.report({'WARNING'}, "No visible cameras found in current View Layer")
            return {'CANCELLED'}


        scene.frame_set(scene.frame_start)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def step(self, context : bpy.types.Context):

        scene = self.scene
        camera = self.cameras[self.current_camera_index]

        print(f"Rendering frame {scene.frame_current}, camera '{camera.name}'")

        ctx = make_format_context(scene, camera)
        raw_name = safe_format(scene.xlr_filename_template.strip(), ctx)
        filename = sanitize_filename(raw_name)

        # render.filepath is base path (no ext); blender adds extension
        
        filepath = os.path.join(resolve_output_dir(self.original_render_filepath), filename)

        scene.camera = camera
        scene.render.filepath = filepath

        # !!!!!! MUST use 'INVOKE_DEFAULT' otherwise camera change won't take effect !!!!!
        # !!!!!! WHY??? Blender API is weird sometimes !!!!!!
        bpy.ops.render.render('INVOKE_DEFAULT', write_still=True, use_viewport=False)

        # next camera / next frame
        self.current_camera_index += 1
        if self.current_camera_index >= len(self.cameras):
            scene.frame_set(scene.frame_current + 1)
            self.current_camera_index = 0

        # progress
        total_tasks = len(self.cameras) * (scene.frame_end - scene.frame_start + 1)
        done = (len(self.cameras) * (scene.frame_current - scene.frame_start)) + self.current_camera_index
        progress = min(max(done / total_tasks, 0.0), 1.0) if total_tasks > 0 else 0.0
        scene.render_progress = progress

    def modal(self, context, event):
        scene = context.scene

        if scene.frame_current > scene.frame_end:
            self.finish(context)
            scene.render_progress = 0.0
            self.report({'INFO'}, "Rendering completed")
            return {'FINISHED'}

        if event.type == 'TIMER':
            self.step(context)

        if event.type == 'ESC':
            self.finish(context)
            scene.render_progress = 0.0
            self.report({'INFO'}, "Rendering cancelled")
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def finish(self, context):
        scene = context.scene
        scene.camera = self.original_camera
        scene.render.filepath = self.original_render_filepath
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None


class XLR_OT_AddTagProperty(Operator):
    bl_idname = "xlr.add_tag_property"
    bl_label = "Add 'Tag' Property to Cameras"

    def execute(self, context):
        for obj in bpy.context.scene.objects:
            if obj.type == 'CAMERA' and "tag" not in obj:
                obj["tag"] = ""
        return {'FINISHED'}

# -------------------------
# Panels
# -------------------------

class XLR_PT_FilePanel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XLR"
    bl_label = "File"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Output
        box = layout.box()

        box.prop(scene.render, "filepath")
        box.label(text=f"Resolved Path: {resolve_output_dir(scene.render.filepath)}", icon='INFO')

        layout.separator()

        # Group: Filename Template
        box = layout.box()
        box.prop(scene, "xlr_filename_template", text="Filename")

        hint = box.row()
        hint.enabled = False  # make it look like a comment
        hint.label(text="Fields: {cam} {frame} {tag} {marker}")

        cam = pick_preview_camera(context)
        ctx = make_format_context(scene, cam) if cam else {
            "cam": "Camera", "frame": scene.frame_current, "tag": "", "marker": ""
        }
        preview = safe_format(scene.xlr_filename_template, ctx)

        box.label(text=f"Preview: {preview}", icon='INFO')



class XLR_PT_CaptionPanel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XLR"
    bl_label = "Caption"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.prop(scene, "xlr_caption_template", text="Caption")

        hint = box.row()
        hint.enabled = False  # make it look like a comment
        hint.label(text="Fields: {cam} {frame} {tag} {marker}")

        cam = pick_preview_camera(context)
        ctx = make_format_context(scene, cam) if cam else {
            "cam": "Camera", "frame": scene.frame_current, "tag": "", "marker": ""
        }
        preview = safe_format_caption(scene.xlr_caption_template, ctx)

        box.label(text=f"Preview: {preview}", icon='INFO')

        layout.separator()
        layout.operator("xlr.generate_captions", icon='FILE_TEXT')



class XLR_PT_RenderPanel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XLR"
    bl_label = "Render"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # area 1: Resolution & Frame Range

        box = layout.box()

        row = box.row()
        row.label(text="Resolution")
        sublayout = row.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=True)
        sublayout.prop(scene.render, "resolution_x", text="X")
        sublayout.prop(scene.render, "resolution_y", text="Y")

        row = box.row()
        row.label(text="Frame Range")
        sublayout = row.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=True)
        sublayout.prop(scene, "frame_start", text="Start")
        sublayout.prop(scene, "frame_end", text="End")

        # area 2: Render Settings

        layout.separator()

        box = layout.box()

        box.label(text="Render Engine:")
        box.prop(scene.render, "engine")

        if scene.render.engine in {'BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'}:
            box.prop(scene.eevee, "taa_render_samples")
        elif scene.render.engine == 'CYCLES':
            box.prop(scene.cycles, "samples")
            box.prop(scene.cycles, "device")

        layout.separator()
        layout.operator(XLR_OT_RenderAllCameras.bl_idname)

        if scene.render_progress > 0:
            layout.separator()
            layout.label(text="Rendering... ESC to stop")
            layout.prop(scene, "render_progress", slider=True)


# -------------------------
# Object["tag"] wrapper
# -------------------------
def _get_tag(obj):
    # 没有属性就返回空串（用于可编辑输入框）
    return obj.get("tag", "")

def _set_tag(obj, value):
    # 一编辑就写入自定义属性（自动创建）
    obj["tag"] = value


class XLR_PT_UtilitiesPanel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XLR"
    bl_label = "Tag"

    def draw(self, context):
        layout = self.layout
        obj = context.view_layer.objects.active

        box = layout.box()

        # 标题：Selected Camera + 当前相机名
        cam_name = obj.name if (obj and obj.type == "CAMERA") else "NONE"
        box.label(text=f"Selected Camera: {cam_name}")

        if not obj or obj.type != "CAMERA":
            box.label(text="No camera selected")
        else:
            # 编辑框（永远可编辑；空串代表还没设置）
            row = box.row(align=True)
            row.label(text="Tag:")
            row.prop(obj, "xlr_tag", text="")

        # 把 Add Tag 按钮放到 box 下面
        layout.operator(XLR_OT_AddTagProperty.bl_idname)



# -------------------------
# Register (your system auto-registers classes; we only register props)
# -------------------------

def register():
    bpy.types.Scene.render_progress = bpy.props.FloatProperty(
        name="Render Progress", default=0.0, min=0.0, max=1.0
    )

    bpy.types.Scene.xlr_filename_template = bpy.props.StringProperty(
        name="Filename Template",
        default="{cam}_{frame:04d}",
        description="Python format string. Fields: cam, frame, tag, marker. Example: {cam}_{frame:04d}"
    )

    bpy.types.Scene.xlr_caption_template = bpy.props.StringProperty(
        name="Caption Template",
        default="{tag}",
        description="Python format string for .txt. Fields: cam, frame, tag, marker. Example: {tag} {marker}"
    )

    bpy.types.Object.xlr_tag = StringProperty(
        name="Tag",
        description='Stored in Object["tag"]',
        get=_get_tag,
        set=_set_tag,
    )

def unregister():
    del bpy.types.Scene.render_progress
    del bpy.types.Scene.xlr_filename_template
    del bpy.types.Scene.xlr_caption_template
    del bpy.types.Object.xlr_tag

if __name__ == "__main__":
    register()
