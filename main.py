import bpy
import os
import re
from bpy.types import Operator, Panel, UIList, PropertyGroup
from bpy.props import (
    StringProperty,
    BoolProperty,
    PointerProperty,
    CollectionProperty,
    IntProperty,
    FloatProperty,
)

# =========================================================
# Constants
# =========================================================

_WINDOWS_FORBIDDEN = r'[<>:"/\\|?*\x00-\x1F]'


# =========================================================
# Small Utils
# =========================================================

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _sanitize_filename(name: str) -> str:
    name = re.sub(_WINDOWS_FORBIDDEN, "_", name).strip().strip(".")
    return name or "untitled"


def _resolve_output_dir(p: str) -> str:
    if not p:
        return bpy.path.abspath("//")
    ap = bpy.path.abspath(p)
    return os.path.abspath(ap)


def _safe_format(template: str, ctx: dict) -> str:
    try:
        return template.format(**ctx)
    except Exception as e:
        return f"[FORMAT ERROR: {e}]"


def _clean_caption(s: str) -> str:
    parts = [p.strip() for p in s.split(",")]
    return ", ".join([p for p in parts if p])


def _format_caption(template: str, ctx: dict) -> str:
    return _clean_caption(_safe_format(template, ctx))


def _pick_preview_camera(context):
    obj = context.object
    if obj and obj.type == "CAMERA":
        return obj

    scene = context.scene
    if scene.camera and scene.camera.type == "CAMERA":
        return scene.camera

    for o in scene.objects:
        if o.type == "CAMERA":
            return o
    return None


def _get_primary_collection_name(obj) -> str:
    try:
        cols = list(obj.users_collection)
    except Exception:
        cols = []
    if not cols:
        return ""
    cols.sort(key=lambda c: (c.name.lower(), c.name))
    return cols[0].name


def _get_marker_name(scene, frame: int) -> str:
    for m in scene.timeline_markers:
        if m.frame == frame:
            return m.name
    return ""


def _make_format_context(scene, cam):
    frame = scene.frame_current
    marker = _get_marker_name(scene, frame)

    if cam:
        return {
            "cam": cam.name,
            "frame": frame,
            "tag": cam.get("tag", ""),
            "marker": marker,
            "collection": _get_primary_collection_name(cam),
        }
    else:
        return {
            "cam": "Camera",
            "frame": frame,
            "tag": "",
            "marker": marker,
            "collection": "",
        }


def _make_rename_context(scene, cam, index: int):
    return {
        "cam": cam.name,
        "tag": cam.get("tag", ""),
        "collection": _get_primary_collection_name(cam),
        "index": index,
    }


def _iter_collection_recursive(coll):
    yield coll
    for ch in coll.children:
        yield from _iter_collection_recursive(ch)


def _collect_cameras_from_collections(colls, include_children=True):
    objs = set()
    for coll in colls:
        if not coll:
            continue
        it = _iter_collection_recursive(coll) if include_children else (coll,)
        for c in it:
            for o in c.objects:
                objs.add(o)

    cams = [o for o in objs if o.type == "CAMERA"]
    return cams


def _collect_listed_cameras(scene):
    items = getattr(scene, "xlr_camera_collections", None)
    if not items:
        return []
    colls = [it.collection for it in items if it.use and it.collection]
    if not colls:
        return []
    return _collect_cameras_from_collections(colls, include_children=True)


def _rename_with_prefix(scene, cameras):
    if not cameras:
        return 0
    for i, cam in enumerate(cameras, start=1):
        prefix = _safe_format(scene.camera_rename_prefix.strip(), _make_rename_context(scene, cam, i))
        cam.name = f"{prefix}{i}"
    return len(cameras)


# =========================================================
# UI Helpers
# =========================================================

def ui_op(layout, op_cls, *, text=None, icon="NONE", **props):
    """
    Draw an operator button using the operator class.
    - Uses op_cls.bl_idname automatically.
    - If text is None, uses op_cls.bl_label.
    - You can pass operator properties via **props.
    """
    if text is None:
        text = getattr(op_cls, "bl_label", "")
    op = layout.operator(op_cls.bl_idname, text=text, icon=icon)
    for k, v in props.items():
        setattr(op, k, v)
    return op


# =========================================================
# Data / UIList
# =========================================================

class XLR_CameraCollectionItem(PropertyGroup):
    use: BoolProperty(name="Use", default=True)
    collection: PointerProperty(name="Collection", type=bpy.types.Collection)


class XLR_UL_CameraCollections(UIList):
    bl_idname = "XLR_UL_camera_collections"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "use", text="")
        row.label(text=(item.collection.name if item.collection else "<None>"), icon="OUTLINER_COLLECTION")
        ui_op(row, XLR_OT_RemoveCameraCollectionAt, text="", icon="X", index=index)


# =========================================================
# Operators
# =========================================================

class XLR_OT_GenerateCaptions(Operator):
    bl_idname = "xlr.generate_captions"
    bl_label = "Generate Captions (.txt)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        out_dir = _resolve_output_dir(scene.render.filepath)

        cams = _collect_listed_cameras(scene)
        if not cams:
            self.report({"WARNING"}, "No cameras found in listed collections")
            return {"CANCELLED"}

        orig_frame = scene.frame_current

        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)
            for cam in cams:
                ctx = _make_format_context(scene, cam)

                filename = _sanitize_filename(_safe_format(scene.xlr_filename_template.strip(), ctx))
                txt_path = os.path.join(out_dir, f"{filename}.txt")
                caption = _format_caption(scene.xlr_caption_template, ctx)

                _ensure_dir(txt_path)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)

        scene.frame_set(orig_frame)
        self.report({"INFO"}, f"Captions generated to: {out_dir}")
        return {"FINISHED"}


class XLR_OT_RenderAllCameras(Operator):
    bl_idname = "xlr.render_all_cameras"
    bl_label = "Render All Cameras"
    bl_options = {"REGISTER", "UNDO"}

    _timer = None

    def invoke(self, context, event):
        self.scene = context.scene
        self.cameras = _collect_listed_cameras(self.scene)
        self.current_camera_index = 0

        if not self.cameras:
            self.report({"WARNING"}, "No cameras found in listed collections")
            return {"CANCELLED"}

        self.original_camera = self.scene.camera
        self.original_render_filepath = self.scene.render.filepath

        self.scene.frame_set(self.scene.frame_start)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def step(self):
        scene = self.scene
        cam = self.cameras[self.current_camera_index]

        ctx = _make_format_context(scene, cam)
        filename = _sanitize_filename(_safe_format(scene.xlr_filename_template.strip(), ctx))
        filepath = os.path.join(_resolve_output_dir(self.original_render_filepath), filename)

        scene.camera = cam
        scene.render.filepath = filepath

        bpy.ops.render.render("INVOKE_DEFAULT", write_still=True, use_viewport=False)

        self.current_camera_index += 1
        if self.current_camera_index >= len(self.cameras):
            scene.frame_set(scene.frame_current + 1)
            self.current_camera_index = 0

        total = len(self.cameras) * (scene.frame_end - scene.frame_start + 1)
        done = (len(self.cameras) * (scene.frame_current - scene.frame_start)) + self.current_camera_index
        scene.render_progress = min(max(done / total, 0.0), 1.0) if total > 0 else 0.0

    def modal(self, context, event):
        scene = context.scene

        if scene.frame_current > scene.frame_end:
            self.finish(context)
            scene.render_progress = 0.0
            self.report({"INFO"}, "Rendering completed")
            return {"FINISHED"}

        if event.type == "TIMER":
            self.step()

        if event.type == "ESC":
            self.finish(context)
            scene.render_progress = 0.0
            self.report({"INFO"}, "Rendering cancelled")
            return {"CANCELLED"}

        return {"PASS_THROUGH"}

    def finish(self, context):
        scene = context.scene
        scene.camera = self.original_camera
        scene.render.filepath = self.original_render_filepath

        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None


class XLR_OT_AddCameraCollection(Operator):
    bl_idname = "xlr.camera_collection_add"
    bl_label = "Add Collection"

    def execute(self, context):
        scene = context.scene

        active_lc = getattr(context.view_layer, "active_layer_collection", None)
        active_coll = active_lc.collection if active_lc else None

        item = scene.xlr_camera_collections.add()
        item.use = True
        item.collection = active_coll

        scene.xlr_camera_collections_index = max(0, len(scene.xlr_camera_collections) - 1)
        return {"FINISHED"}


class XLR_OT_RemoveCameraCollection(Operator):
    bl_idname = "xlr.camera_collection_remove"
    bl_label = "Remove Collection"

    def execute(self, context):
        scene = context.scene
        idx = scene.xlr_camera_collections_index
        if 0 <= idx < len(scene.xlr_camera_collections):
            scene.xlr_camera_collections.remove(idx)
            scene.xlr_camera_collections_index = min(idx, len(scene.xlr_camera_collections) - 1)
        return {"FINISHED"}


class XLR_OT_RemoveCameraCollectionAt(Operator):
    bl_idname = "xlr.camera_collection_remove_at"
    bl_label = "Remove Collection (At Index)"

    index: IntProperty(default=-1)

    def execute(self, context):
        scene = context.scene
        idx = self.index
        if 0 <= idx < len(scene.xlr_camera_collections):
            scene.xlr_camera_collections.remove(idx)
            scene.xlr_camera_collections_index = min(
                scene.xlr_camera_collections_index,
                len(scene.xlr_camera_collections) - 1,
            )
            return {"FINISHED"}
        return {"CANCELLED"}


class XLR_OT_RenameAllCameras(Operator):
    bl_idname = "xlr.rename_all_cameras"
    bl_label = "Rename All"

    def execute(self, context):
        scene = context.scene
        n = _rename_with_prefix(scene, _collect_listed_cameras(scene))
        if n == 0:
            self.report({"WARNING"}, "No cameras found in listed collections")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Renamed {n} cameras")
        return {"FINISHED"}


class XLR_OT_RenameInActiveCollection(Operator):
    bl_idname = "xlr.rename_camera_in_active_collection"
    bl_label = "Rename In Active Collection"

    def execute(self, context):
        scene = context.scene
        idx = scene.xlr_camera_collections_index

        if not (0 <= idx < len(scene.xlr_camera_collections)):
            self.report({"WARNING"}, "No active collection item")
            return {"CANCELLED"}

        coll = scene.xlr_camera_collections[idx].collection
        if not coll:
            self.report({"WARNING"}, "Active item has no collection set")
            return {"CANCELLED"}

        n = _rename_with_prefix(scene, _collect_cameras_from_collections([coll], include_children=True))
        if n == 0:
            self.report({"WARNING"}, f"No cameras found in: {coll.name}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Renamed {n} cameras in: {coll.name}")
        return {"FINISHED"}


# =========================================================
# Tag Property Wrapper (Object["tag"])
# =========================================================

def _get_tag(obj):
    return obj.get("tag", "")

def _set_tag(obj, value):
    obj["tag"] = value


# =========================================================
# Panels
# =========================================================

class XLR_PT_FilePanel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XLR"
    bl_label = "File"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.prop(scene.render, "filepath")
        box.label(text=f"Resolved Path: {_resolve_output_dir(scene.render.filepath)}", icon="INFO")

        box = layout.box()
        box.prop(scene, "xlr_filename_template", text="Filename")
        hint = box.row()
        hint.enabled = False
        hint.label(text="Fields: {cam} {frame} {tag} {marker} {collection}")

        cam = _pick_preview_camera(context)
        box.label(text=f"Preview: {_safe_format(scene.xlr_filename_template, _make_format_context(scene, cam))}", icon="INFO")


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
        hint.enabled = False
        hint.label(text="Fields: {cam} {frame} {tag} {marker} {collection}")

        cam = _pick_preview_camera(context)
        box.label(text=f"Preview: {_format_caption(scene.xlr_caption_template, _make_format_context(scene, cam))}", icon="INFO")

        layout.separator()
        ui_op(layout, XLR_OT_GenerateCaptions, icon="FILE_TEXT")


class XLR_PT_RenderPanel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XLR"
    bl_label = "Render"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        row = box.row()
        row.label(text="Resolution")
        sub = row.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=True)
        sub.prop(scene.render, "resolution_x", text="X")
        sub.prop(scene.render, "resolution_y", text="Y")

        row = box.row()
        row.label(text="Frame Range")
        sub = row.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=True)
        sub.prop(scene, "frame_start", text="Start")
        sub.prop(scene, "frame_end", text="End")

        box = layout.box()
        box.label(text="Render Engine:")
        box.prop(scene.render, "engine")
        if scene.render.engine in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
            box.prop(scene.eevee, "taa_render_samples")
        elif scene.render.engine == "CYCLES":
            box.prop(scene.cycles, "samples")
            box.prop(scene.cycles, "device")

        layout.separator()
        ui_op(layout, XLR_OT_RenderAllCameras)
        if scene.render_progress > 0:
            layout.separator()
            layout.label(text="Rendering... ESC to stop")
            layout.prop(scene, "render_progress", slider=True)


class XLR_PT_CameraPanel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XLR"
    bl_label = "Camera"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.view_layer.objects.active

        box = layout.box()
        box.label(text="Collections")

        row = box.row()
        row.template_list(
            "XLR_UL_camera_collections", "",
            scene, "xlr_camera_collections",
            scene, "xlr_camera_collections_index",
            rows=2
        )
        col = row.column(align=True)
        ui_op(col, XLR_OT_AddCameraCollection, text="", icon="ADD")
        ui_op(col, XLR_OT_RemoveCameraCollection, text="", icon="REMOVE")

        idx = scene.xlr_camera_collections_index
        if 0 <= idx < len(scene.xlr_camera_collections):
            it = scene.xlr_camera_collections[idx]
            box.prop_search(it, "collection", bpy.data, "collections", text="Collection")
        else:
            box.label(text="No item selected")

        layout.separator()

        box = layout.box()
        cam_name = obj.name if (obj and obj.type == "CAMERA") else "NONE"
        box.label(text=f"Selected Camera: {cam_name}")
        if obj and obj.type == "CAMERA":
            box.prop(obj, "xlr_tag", text="Tag")
        else:
            box.label(text="No camera selected")

        layout.separator()

        box = layout.box()
        box.label(text="Camera Renaming")
        box.prop(scene, "camera_rename_prefix", text="prefix")

        hint = box.row()
        hint.enabled = False
        hint.label(text="Fields: {cam} {tag} {collection} {index}")

        if 0 <= idx < len(scene.xlr_camera_collections):
            it = scene.xlr_camera_collections[idx]
            active_name = it.collection.name if it.collection else "<None>"
        else:
            active_name = "<None>"

        row = box.row(align=True)
        ui_op(row, XLR_OT_RenameAllCameras)
        ui_op(row, XLR_OT_RenameInActiveCollection, text=f"Rename in {active_name}")


# =========================================================
# Register / Unregister
# =========================================================

def register():
    bpy.types.Scene.render_progress = FloatProperty(name="Render Progress", default=0.0, min=0.0, max=1.0)

    bpy.types.Scene.xlr_filename_template = StringProperty(
        name="Filename Template",
        default="{cam}_{frame:04d}",
        description="Fields: cam, frame, tag, marker, collection.",
    )
    bpy.types.Scene.xlr_caption_template = StringProperty(
        name="Caption Template",
        default="{tag}",
        description="Fields: cam, frame, tag, marker, collection.",
    )

    bpy.types.Scene.xlr_camera_collections = CollectionProperty(type=XLR_CameraCollectionItem)
    bpy.types.Scene.xlr_camera_collections_index = IntProperty(default=0)

    bpy.types.Scene.camera_rename_prefix = StringProperty(
        name="Camera Rename Prefix",
        default="{collection}_",
        description="Fields: cam, tag, collection, index.",
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
    del bpy.types.Scene.xlr_camera_collections
    del bpy.types.Scene.xlr_camera_collections_index
    del bpy.types.Scene.camera_rename_prefix
    del bpy.types.Object.xlr_tag


if __name__ == "__main__":
    register()
