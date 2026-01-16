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

import tempfile
import uuid
from math import pi
from mathutils import Vector, Euler


# =========================================================
# Constants
# =========================================================

_WINDOWS_FORBIDDEN = r'[<>:"/\\|?*\x00-\x1F]'


# =========================================================
# Formatting (STRICT)
# =========================================================

class XLRFormatError(Exception):
    pass


def _format_strict(template: str, ctx: dict, *, template_name: str = "template") -> str:
    """
    Strict format:
    - Unknown field -> raises XLRFormatError with clear message
    - Any format error -> raises XLRFormatError
    """
    try:
        return template.format(**ctx)
    except KeyError as e:
        key = e.args[0] if e.args else "<unknown>"
        raise XLRFormatError(f"Unknown field {{{key}}} in {template_name}: {template!r}")
    except Exception as e:
        raise XLRFormatError(f"Format error in {template_name}: {template!r} -> {e}")


def _format_preview(template: str, ctx: dict, *, template_name: str = "template") -> str:
    """
    Preview helper for UI only: never throws; returns an error string.
    """
    try:
        return _format_strict(template, ctx, template_name=template_name)
    except Exception as e:
        return f"[FORMAT ERROR: {e}]"


# =========================================================
# Small Utils
# =========================================================

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _sanitize_filename(name: str) -> str:
    name = re.sub(_WINDOWS_FORBIDDEN, "_", name).strip().strip(".")
    return name or "untitled"


def _sanitize_relpath(p: str) -> str:
    """
    Sanitize a relative path that may contain folders like "A/B/C".
    - Split by both '/' and '\\'
    - Sanitize each segment with _sanitize_filename
    - Re-join with OS separator
    """
    if not p:
        return "untitled"
    parts = re.split(r"[\\/]+", p.strip())
    parts = [_sanitize_filename(x) for x in parts if x.strip()]
    return os.path.join(*parts) if parts else "untitled"


def _resolve_output_dir(p: str) -> str:
    if not p:
        return bpy.path.abspath("//")
    ap = bpy.path.abspath(p)
    return os.path.abspath(ap)


def _clean_caption(s: str) -> str:
    parts = [p.strip() for p in s.split(",")]
    return ", ".join([p for p in parts if p])


def _format_caption_strict(template: str, ctx: dict) -> str:
    return _clean_caption(_format_strict(template, ctx, template_name="Caption Template"))


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

def _get_current_marker_name(scene) -> str:
    return _get_marker_name(scene, scene.frame_current)


def _get_active_env_item(scene):
    items = getattr(scene, "xlr_envs", None)
    if not items:
        return None
    idx = getattr(scene, "xlr_envs_index", -1)
    if not (0 <= idx < len(items)):
        return None
    return items[idx]


def _get_active_env_name(scene) -> str:
    it = _get_active_env_item(scene)
    if not it:
        return ""
    return it.collection.name if it.collection else ""


def _get_active_env_tag(scene) -> str:
    it = _get_active_env_item(scene)
    if not it:
        return ""
    return getattr(it, "tag", "") or ""


def _find_first_camera_in_collection_recursive(coll: bpy.types.Collection):
    """Depth-first, early-exit: return first camera object found, else None."""
    if not coll:
        return None
    # objects first
    for o in coll.objects:
        if o and o.type == "CAMERA":
            return o
    # then children
    for ch in coll.children:
        hit = _find_first_camera_in_collection_recursive(ch)
        if hit:
            return hit
    return None


def _find_first_listed_camera(scene: bpy.types.Scene):
    """Early-exit: return first camera in xlr_camera_collections list (use=True), else None."""
    items = getattr(scene, "xlr_camera_collections", None)
    if not items:
        return None
    for it in items:
        if not getattr(it, "use", False):
            continue
        coll = getattr(it, "collection", None)
        hit = _find_first_camera_in_collection_recursive(coll)
        if hit:
            return hit
    return None


def _pick_preview_camera(context):
    obj = getattr(context, "object", None)
    if obj and obj.type == "CAMERA":
        return obj
    return _find_first_listed_camera(context.scene)


_NONE = "<NONE>"

def _make_format_context(scene, cam):
    def _cam_field(value):
        return value if value not in (None, "") else _NONE

    is_cam = bool(cam) and getattr(cam, "type", None) == "CAMERA"

    cam_name = _cam_field(cam.name) if is_cam else _NONE
    cam_tag  = _cam_field(cam.get("tag", "")) if is_cam else _NONE
    cam_coll = _cam_field(_get_primary_collection_name(cam)) if is_cam else _NONE

    return {
        "cam": cam_name,
        "frame": scene.frame_current,
        "cam_tag": cam_tag,
        "marker": _get_current_marker_name(scene),
        "cam_coll": cam_coll,
        "env": _get_active_env_name(scene),
        "env_tag": _get_active_env_tag(scene),
    }




def _make_rename_context(scene, cam, index: int):
    cam_coll = _get_primary_collection_name(cam) if cam else ""
    return {
        "cam": cam.name,
        "cam_tag": cam.get("tag", ""),
        "cam_coll": cam_coll,
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


def _rename_with_prefix(scene, cameras, reporter=None):
    """
    Strict rename:
    - if prefix template has unknown field -> skip that camera (and report)
    """
    if not cameras:
        return 0

    ok_count = 0
    for i, cam in enumerate(cameras, start=1):
        try:
            prefix = _format_strict(scene.camera_rename_prefix.strip(), _make_rename_context(scene, cam, i),
                                  template_name="Camera Rename Prefix")
        except Exception as e:
            if reporter:
                reporter({"WARNING"}, f"Rename skipped for {cam.name}: {e}")
            continue

        cam.name = f"{prefix}{i}"
        ok_count += 1

    return ok_count


# =========================================================
# World Bake Helpers
# =========================================================

def _remove_world_if_exists(name: str):
    w = bpy.data.worlds.get(name)
    if w:
        try:
            w.use_fake_user = False
        except Exception:
            pass
        bpy.data.worlds.remove(w, do_unlink=True)


def _remove_image_if_exists(name: str):
    img = bpy.data.images.get(name)
    if img:
        try:
            img.use_fake_user = False
        except Exception:
            pass
        bpy.data.images.remove(img, do_unlink=True)


def _disable_local_cameras_in_window(context):
    """
    Disable use_local_camera for all VIEW_3D spaces in current window/screen.
    Returns a restore token list: [(space, prev_bool), ...]
    """
    token = []
    win = getattr(context, "window", None)
    scr = win.screen if win else None
    if not scr:
        return token

    for area in scr.areas:
        if area.type != "VIEW_3D":
            continue
        for space in area.spaces:
            if space.type != "VIEW_3D":
                continue
            try:
                prev = bool(space.use_local_camera)
                token.append((space, prev))
                if prev:
                    space.use_local_camera = False
            except Exception:
                pass
    return token


def _restore_local_cameras(token):
    for space, prev in token:
        try:
            space.use_local_camera = prev
        except Exception:
            pass


def _snapshot_render_settings(scene):
    r = scene.render
    img = r.image_settings
    cycles_samples = None
    if hasattr(scene, "cycles") and hasattr(scene.cycles, "samples"):
        cycles_samples = scene.cycles.samples
    return {
        "camera": scene.camera,
        "world": scene.world,
        "engine": r.engine,
        "filepath": r.filepath,
        "res_x": r.resolution_x,
        "res_y": r.resolution_y,
        "res_pct": r.resolution_percentage,
        "file_format": img.file_format,
        "color_mode": img.color_mode,
        "color_depth": img.color_depth,
        "cycles_samples": cycles_samples,
    }


def _restore_render_settings(scene, snap):
    r = scene.render
    img = r.image_settings
    scene.camera = snap["camera"]
    scene.world = snap["world"]
    r.engine = snap["engine"]
    r.filepath = snap["filepath"]
    r.resolution_x = snap["res_x"]
    r.resolution_y = snap["res_y"]
    r.resolution_percentage = snap["res_pct"]
    img.file_format = snap["file_format"]
    img.color_mode = snap["color_mode"]
    img.color_depth = snap["color_depth"]
    if snap["cycles_samples"] is not None and hasattr(scene, "cycles") and hasattr(scene.cycles, "samples"):
        scene.cycles.samples = snap["cycles_samples"]


def _snapshot_collection_hide_render(scene):
    state = {}
    for col in scene.collection.children:
        state[col.name_full] = bool(col.hide_render)
    return state


def _apply_collection_hide_render(scene, state, hide_all=True, keep_collection=None):
    for col in scene.collection.children:
        if keep_collection and col == keep_collection:
            continue
        if hide_all:
            col.hide_render = True
        else:
            col.hide_render = state.get(col.name_full, False)


def _bake_world_to_env_texture(
    context,
    *,
    resolution: int = 2048,
    cycles_samples: int = 4,
    file_format: str = "HDR",   # "HDR" / "OPEN_EXR" / "PNG"
) -> tuple[bpy.types.Image | None, str | None]:

    scene = context.scene
    snap = _snapshot_render_settings(scene)

    bake_camera_data = bpy.data.cameras.new(".XLR_BakeCamera")
    bake_camera_data.type = "PANO"
    if hasattr(bake_camera_data, "cycles"):
        bake_camera_data.cycles.panorama_type = "EQUIRECTANGULAR"
    else:
        bake_camera_data.panorama_type = "EQUIRECTANGULAR"

    bake_camera = bpy.data.objects.new(".XLR_BakeCamera", bake_camera_data)
    bake_camera.location = Vector((0, 0, 0))
    bake_camera.rotation_euler = Euler((pi / 2.0, 0, -pi / 2.0))

    bake_coll = bpy.data.collections.new(".XLR_Bake")
    scene.collection.children.link(bake_coll)
    bake_coll.objects.link(bake_camera)

    hide_state = _snapshot_collection_hide_render(scene)
    _apply_collection_hide_render(scene, hide_state, hide_all=True, keep_collection=bake_coll)

    tmp_path = None
    loaded_img = None

    try:
        scene.render.engine = "CYCLES"
        scene.camera = bake_camera

        scene.render.resolution_x = int(resolution)
        scene.render.resolution_y = int(resolution // 2)
        scene.render.resolution_percentage = 100

        scene.render.image_settings.file_format = file_format
        scene.render.image_settings.color_mode = "RGB"
        if file_format in {"HDR", "OPEN_EXR"}:
            scene.render.image_settings.color_depth = "32"
        elif file_format == "PNG":
            scene.render.image_settings.color_depth = "16"

        if hasattr(scene, "cycles") and hasattr(scene.cycles, "samples"):
            scene.cycles.samples = int(cycles_samples)

        bpy.ops.render.render()

        rr = bpy.data.images.get("Render Result")
        if rr is None:
            return None, None

        ext = ".hdr"
        if file_format == "OPEN_EXR":
            ext = ".exr"
        elif file_format == "PNG":
            ext = ".png"

        tmp_name = f"XLR_WorldBake_{uuid.uuid4().hex}{ext}"
        tmp_path = os.path.join(tempfile.gettempdir(), tmp_name)

        rr.save_render(filepath=tmp_path)
        loaded_img = bpy.data.images.load(tmp_path, check_existing=False)

    finally:
        _apply_collection_hide_render(scene, hide_state, hide_all=False)

        try:
            bpy.data.objects.remove(bake_camera, do_unlink=True)
        except Exception:
            pass
        try:
            bpy.data.cameras.remove(bake_camera_data, do_unlink=True)
        except Exception:
            pass
        try:
            bpy.data.collections.remove(bake_coll)
        except Exception:
            pass

        _restore_render_settings(scene, snap)

    return loaded_img, tmp_path


def _build_baked_world(world_name: str, baked_img: bpy.types.Image):
    w = bpy.data.worlds.new(world_name)
    w.use_nodes = True
    nt = w.node_tree
    nodes = nt.nodes
    links = nt.links

    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputWorld")
    out.location = (400, 0)

    bg = nodes.new("ShaderNodeBackground")
    bg.location = (150, 0)

    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.location = (-200, 0)
    env_tex.image = baked_img

    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])
    return w


# =========================================================
# ViewLayer / Exclude Helpers (Env)
# =========================================================

def _find_layer_collection(layer_coll: bpy.types.LayerCollection, target_coll: bpy.types.Collection):
    if not layer_coll or not target_coll:
        return None
    if layer_coll.collection == target_coll:
        return layer_coll
    for ch in layer_coll.children:
        found = _find_layer_collection(ch, target_coll)
        if found:
            return found
    return None


def _set_collection_exclude(view_layer: bpy.types.ViewLayer, coll: bpy.types.Collection, excluded: bool) -> bool:
    if not view_layer or not coll:
        return False
    root = view_layer.layer_collection
    lc = _find_layer_collection(root, coll)
    if not lc:
        return False
    lc.exclude = bool(excluded)
    return True


def _activate_env(scene: bpy.types.Scene, view_layer: bpy.types.ViewLayer, env_index: int) -> bool:
    items = getattr(scene, "xlr_envs", None)
    if not items or not (0 <= env_index < len(items)):
        return False

    active = items[env_index]
    if not active.collection:
        return False

    if active.world:
        scene.world = active.world

    for i, it in enumerate(items):
        if not it.collection:
            continue
        _set_collection_exclude(view_layer, it.collection, excluded=(i != env_index))

    scene.xlr_envs_index = env_index
    return True


def _iter_enabled_env_indices(scene: bpy.types.Scene):
    items = getattr(scene, "xlr_envs", None)
    if not items:
        return []
    out = []
    for i, it in enumerate(items):
        if not it.use:
            continue
        if not it.collection:
            continue
        out.append(i)
    return out


# =========================================================
# UI Helpers
# =========================================================

def ui_op(layout, op_cls, *, text=None, icon="NONE", **props):
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


class XLR_EnvItem(PropertyGroup):
    use: BoolProperty(name="Use", default=True)
    collection: PointerProperty(name="Collection", type=bpy.types.Collection)
    world: PointerProperty(name="World", type=bpy.types.World)
    tag: StringProperty(name="Tag", default="", description="Env tag used as {env_tag} replacement field.")


class XLR_UL_CameraCollections(UIList):
    bl_idname = "XLR_UL_camera_collections"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "use", text="")
        row.label(text=(item.collection.name if item.collection else "<None>"), icon="OUTLINER_COLLECTION")
        ui_op(row, XLR_OT_RemoveCameraCollectionAt, text="", icon="X", index=index)


class XLR_UL_Envs(UIList):
    bl_idname = "XLR_UL_envs"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "use", text="")
        row.label(text=(item.collection.name if item.collection else "<None>"), icon="OUTLINER_COLLECTION")
        ui_op(row, XLR_OT_RemoveEnvAt, text="", icon="X", index=index)


# =========================================================
# Operators
# =========================================================

class XLR_OT_GenerateCaptions(Operator):
    bl_idname = "xlr.generate_captions"
    bl_label = "Generate Captions (.txt)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        view_layer = context.view_layer
        out_dir = _resolve_output_dir(scene.render.filepath)

        cams = _collect_listed_cameras(scene)
        if not cams:
            self.report({"WARNING"}, "No cameras found in listed collections")
            return {"CANCELLED"}

        env_indices = _iter_enabled_env_indices(scene)
        if not env_indices:
            env_indices = [-1]

        orig_frame = scene.frame_current
        orig_world = scene.world

        orig_excludes = {}
        if getattr(scene, "xlr_envs", None):
            for it in scene.xlr_envs:
                if not it.collection:
                    continue
                lc = _find_layer_collection(view_layer.layer_collection, it.collection)
                if lc:
                    orig_excludes[it.collection.name_full] = lc.exclude

        skipped = 0
        wrote = 0

        try:
            for env_i in env_indices:
                if env_i >= 0:
                    ok = _activate_env(scene, view_layer, env_i)
                    if not ok:
                        continue

                for frame in range(scene.frame_start, scene.frame_end + 1):
                    scene.frame_set(frame)

                    for cam in cams:
                        ctx = _make_format_context(scene, cam)

                        # strict filename
                        try:
                            raw_name = _format_strict(scene.xlr_filename_template.strip(), ctx,
                                                      template_name="Filename Template")
                        except Exception as e:
                            skipped += 1
                            self.report({"WARNING"}, f"Skip caption (filename template error): {e}")
                            continue

                        filename = _sanitize_relpath(raw_name)
                        txt_path = os.path.join(out_dir, f"{filename}.txt")

                        # strict caption
                        try:
                            caption = _format_caption_strict(scene.xlr_caption_template, ctx)
                        except Exception as e:
                            skipped += 1
                            self.report({"WARNING"}, f"Skip caption (caption template error): {e}")
                            continue

                        _ensure_dir(txt_path)
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(caption)
                        wrote += 1

        finally:
            scene.frame_set(orig_frame)
            scene.world = orig_world

            if getattr(scene, "xlr_envs", None):
                for it in scene.xlr_envs:
                    if not it.collection:
                        continue
                    lc = _find_layer_collection(view_layer.layer_collection, it.collection)
                    if not lc:
                        continue
                    key = it.collection.name_full
                    if key in orig_excludes:
                        lc.exclude = orig_excludes[key]

        self.report({"INFO"}, f"Captions: wrote {wrote}, skipped {skipped}. Output: {out_dir}")
        return {"FINISHED"}


class XLR_OT_RenderAllCameras(Operator):
    bl_idname = "xlr.render_all_cameras"
    bl_label = "Render All Cameras"
    bl_options = {"REGISTER", "UNDO"}

    _timer = None

    def invoke(self, context, event):
        self.scene = context.scene
        self.view_layer = context.view_layer

        self.cameras = _collect_listed_cameras(self.scene)
        if not self.cameras:
            self.report({"WARNING"}, "No cameras found in listed collections")
            return {"CANCELLED"}

        self.env_indices = _iter_enabled_env_indices(self.scene)
        if not self.env_indices:
            self.env_indices = [-1]

        self.frames = list(range(self.scene.frame_start, self.scene.frame_end + 1))
        if not self.frames:
            self.report({"WARNING"}, "Invalid frame range")
            return {"CANCELLED"}

        self.env_i = 0
        self.frame_i = 0
        self.cam_i = 0

        self.original_camera = self.scene.camera
        self.original_render_filepath = self.scene.render.filepath
        self.original_world = self.scene.world

        self.original_excludes = {}
        if getattr(self.scene, "xlr_envs", None):
            for it in self.scene.xlr_envs:
                if not it.collection:
                    continue
                lc = _find_layer_collection(self.view_layer.layer_collection, it.collection)
                if lc:
                    self.original_excludes[it.collection.name_full] = lc.exclude

        self.skipped = 0
        self.rendered = 0

        self._ensure_env_active()
        self.scene.frame_set(self.frames[self.frame_i])

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _ensure_env_active(self):
        if self.env_i >= len(self.env_indices):
            return
        env_idx = self.env_indices[self.env_i]
        if env_idx >= 0:
            _activate_env(self.scene, self.view_layer, env_idx)

    def _is_done(self) -> bool:
        return self.env_i >= len(self.env_indices)

    def _advance(self):
        self.cam_i += 1
        if self.cam_i >= len(self.cameras):
            self.cam_i = 0
            self.frame_i += 1
            if self.frame_i >= len(self.frames):
                self.frame_i = 0
                self.env_i += 1

        total = len(self.env_indices) * len(self.frames) * len(self.cameras)
        done = (
            (self.env_i * len(self.frames) * len(self.cameras)) +
            (self.frame_i * len(self.cameras)) +
            self.cam_i
        )
        self.scene.render_progress = min(max(done / total, 0.0), 1.0) if total > 0 else 0.0

    def step(self):
        if self._is_done():
            return

        scene = self.scene
        self._ensure_env_active()

        frame = self.frames[self.frame_i]
        cam = self.cameras[self.cam_i]

        scene.frame_set(frame)
        ctx = _make_format_context(scene, cam)

        # strict filename
        try:
            raw_name = _format_strict(scene.xlr_filename_template.strip(), ctx, template_name="Filename Template")
        except Exception as e:
            self.skipped += 1
            self.report({"WARNING"}, f"Skip render (filename template error): {e}")
            self._advance()
            return

        filename = _sanitize_relpath(raw_name)
        filepath = os.path.join(_resolve_output_dir(self.original_render_filepath), filename)

        scene.camera = cam
        scene.render.filepath = filepath
        bpy.ops.render.render("INVOKE_DEFAULT", write_still=True, use_viewport=False)
        self.rendered += 1

        self._advance()

    def modal(self, context, event):
        scene = context.scene

        if self._is_done():
            self.finish(context)
            scene.render_progress = 0.0
            self.report({"INFO"}, f"Rendering completed. Rendered {self.rendered}, skipped {self.skipped}.")
            return {"FINISHED"}

        if event.type == "TIMER":
            self.step()

        if event.type == "ESC":
            self.finish(context)
            scene.render_progress = 0.0
            self.report({"INFO"}, f"Rendering cancelled. Rendered {self.rendered}, skipped {self.skipped}.")
            return {"CANCELLED"}

        return {"PASS_THROUGH"}

    def finish(self, context):
        scene = context.scene

        scene.camera = self.original_camera
        scene.render.filepath = self.original_render_filepath
        scene.world = self.original_world

        if getattr(scene, "xlr_envs", None):
            for it in scene.xlr_envs:
                if not it.collection:
                    continue
                lc = _find_layer_collection(self.view_layer.layer_collection, it.collection)
                if not lc:
                    continue
                key = it.collection.name_full
                if key in self.original_excludes:
                    lc.exclude = self.original_excludes[key]

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


class XLR_OT_AddEnv(Operator):
    bl_idname = "xlr.env_add"
    bl_label = "Add Env"

    def execute(self, context):
        scene = context.scene
        active_lc = getattr(context.view_layer, "active_layer_collection", None)
        active_coll = active_lc.collection if active_lc else None

        item = scene.xlr_envs.add()
        item.use = True
        item.collection = active_coll
        item.world = scene.world
        item.tag = ""

        scene.xlr_envs_index = max(0, len(scene.xlr_envs) - 1)
        return {"FINISHED"}


class XLR_OT_RemoveEnv(Operator):
    bl_idname = "xlr.env_remove"
    bl_label = "Remove Env"

    def execute(self, context):
        scene = context.scene
        idx = scene.xlr_envs_index
        if 0 <= idx < len(scene.xlr_envs):
            scene.xlr_envs.remove(idx)
            scene.xlr_envs_index = min(idx, len(scene.xlr_envs) - 1)
        return {"FINISHED"}


class XLR_OT_RemoveEnvAt(Operator):
    bl_idname = "xlr.env_remove_at"
    bl_label = "Remove Env (At Index)"

    index: IntProperty(default=-1)

    def execute(self, context):
        scene = context.scene
        idx = self.index
        if 0 <= idx < len(scene.xlr_envs):
            scene.xlr_envs.remove(idx)
            scene.xlr_envs_index = min(scene.xlr_envs_index, len(scene.xlr_envs) - 1)
            return {"FINISHED"}
        return {"CANCELLED"}


class XLR_OT_ActivateEnv(Operator):
    bl_idname = "xlr.env_activate"
    bl_label = "Activate Env"

    index: IntProperty(default=-1)

    def execute(self, context):
        scene = context.scene
        view_layer = context.view_layer

        idx = self.index if self.index >= 0 else scene.xlr_envs_index
        if not (0 <= idx < len(scene.xlr_envs)):
            self.report({"WARNING"}, "No env selected")
            return {"CANCELLED"}

        ok = _activate_env(scene, view_layer, idx)
        if not ok:
            self.report({"WARNING"}, "Env activation failed (need collection)")
            return {"CANCELLED"}

        name = _get_active_env_name(scene)
        self.report({"INFO"}, f"Env activated: {name}")
        return {"FINISHED"}


class XLR_OT_RenameAllCameras(Operator):
    bl_idname = "xlr.rename_all_cameras"
    bl_label = "Rename All"

    def execute(self, context):
        scene = context.scene
        n = _rename_with_prefix(scene, _collect_listed_cameras(scene), reporter=self.report)
        if n == 0:
            self.report({"WARNING"}, "No cameras renamed (or none found)")
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

        n = _rename_with_prefix(scene, _collect_cameras_from_collections([coll], include_children=True), reporter=self.report)
        if n == 0:
            self.report({"WARNING"}, f"No cameras renamed in: {coll.name}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Renamed {n} cameras in: {coll.name}")
        return {"FINISHED"}


class XLR_OT_BakeActiveEnvWorld(Operator):
    bl_idname = "xlr.env_bake_active_world"
    bl_label = "Bake World to Texture (Active Env)"
    bl_options = {"REGISTER", "UNDO"}

    bake_resolution: IntProperty(name="Resolution", default=2048, min=256, max=16384)
    bake_samples: IntProperty(name="Cycles Samples", default=4, min=1, max=1024)

    def execute(self, context):
        scene = context.scene
        view_layer = context.view_layer

        idx = getattr(scene, "xlr_envs_index", -1)
        if not (0 <= idx < len(scene.xlr_envs)):
            self.report({"WARNING"}, "No env selected")
            return {"CANCELLED"}

        if not _activate_env(scene, view_layer, idx):
            self.report({"WARNING"}, "Env activation failed (need collection)")
            return {"CANCELLED"}

        it = scene.xlr_envs[idx]
        if not scene.world:
            self.report({"WARNING"}, "Scene has no active world to bake")
            return {"CANCELLED"}

        env_name = _get_active_env_name(scene) or (it.collection.name if it.collection else "Env")
        baked_name = f"{env_name}_world_baked"

        _remove_world_if_exists(baked_name)
        _remove_image_if_exists(baked_name)

        local_cam_token = _disable_local_cameras_in_window(context)

        img = None
        tmp_path = None
        try:
            img, tmp_path = _bake_world_to_env_texture(
                context,
                resolution=self.bake_resolution,
                cycles_samples=self.bake_samples,
                file_format="HDR",
            )
        finally:
            _restore_local_cameras(local_cam_token)

        if img is None:
            self.report({"WARNING"}, "Bake failed (no image)")
            return {"CANCELLED"}

        img.name = baked_name

        try:
            img.pack()
        except Exception:
            pass
        for attr in ("filepath", "filepath_raw"):
            if hasattr(img, attr):
                try:
                    setattr(img, attr, "")
                except Exception:
                    pass

        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        baked_world = _build_baked_world(baked_name, img)
        it.world = baked_world
        scene.world = baked_world

        self.report({"INFO"}, f"Baked world applied: {baked_world.name} (Image: {img.name})")
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
        hint.label(text="Fields: {cam} {frame} {cam_tag} {marker} {cam_coll} {env} {env_tag}")

        cam = _pick_preview_camera(context)
        box.label(
            text=f"Preview: {_format_preview(scene.xlr_filename_template, _make_format_context(scene, cam), template_name='Filename Template')}",
            icon="INFO"
        )


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
        hint.label(text="Fields: {cam} {frame} {cam_tag} {marker} {cam_coll} {env} {env_tag}")

        cam = _pick_preview_camera(context)
        # preview = strict format then clean caption for display
        raw = _format_preview(scene.xlr_caption_template, _make_format_context(scene, cam), template_name="Caption Template")
        box.label(text=f"Preview: {_clean_caption(raw)}", icon="INFO")

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
        box.label(text=f"Selected: {cam_name}")
        if obj and obj.type == "CAMERA":
            box.prop(obj, "xlr_tag", text="Tag")
        else:
            box.label(text="No Tag (select a camera)")

        layout.separator()

        box = layout.box()
        box.label(text="Renaming")
        box.prop(scene, "camera_rename_prefix", text="prefix")

        hint = box.row()
        hint.enabled = False
        hint.label(text="Fields: {cam} {cam_tag} {cam_coll} {index}")

        if 0 <= idx < len(scene.xlr_camera_collections):
            it = scene.xlr_camera_collections[idx]
            active_name = it.collection.name if it.collection else "<None>"
        else:
            active_name = "<None>"

        row = box.row(align=True)
        ui_op(row, XLR_OT_RenameAllCameras)
        ui_op(row, XLR_OT_RenameInActiveCollection, text=f"Rename in {active_name}")


class XLR_PT_EnvPanel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XLR"
    bl_label = "Env"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.label(text="Environments (Collection + World)")

        row = box.row()
        row.template_list(
            "XLR_UL_envs", "",
            scene, "xlr_envs",
            scene, "xlr_envs_index",
            rows=3
        )
        col = row.column(align=True)
        ui_op(col, XLR_OT_AddEnv, text="", icon="ADD")
        ui_op(col, XLR_OT_RemoveEnv, text="", icon="REMOVE")

        idx = scene.xlr_envs_index
        if 0 <= idx < len(scene.xlr_envs):
            it = scene.xlr_envs[idx]
            edit = box.box()

            edit.prop_search(it, "collection", bpy.data, "collections", text="Col")
            edit.prop_search(it, "world", bpy.data, "worlds", text="World")
            edit.prop(it, "tag", text="Tag")

            row = edit.row(align=True)
            ui_op(row, XLR_OT_ActivateEnv, text="Activate", icon="CHECKMARK")

            op = ui_op(edit, XLR_OT_BakeActiveEnvWorld, text="Bake World", icon="RENDER_STILL")

            r = edit.row(align=True)
            r.prop(scene, "xlr_world_bake_resolution", text="Resolution")
            r.prop(scene, "xlr_world_bake_samples", text="Samples")

            op.bake_resolution = scene.xlr_world_bake_resolution
            op.bake_samples = scene.xlr_world_bake_samples

        else:
            box.label(text="No env selected")


# =========================================================
# Register / Unregister (types are your responsibility; props are mine)
# =========================================================

def register():
    bpy.types.Scene.render_progress = FloatProperty(name="Render Progress", default=0.0, min=0.0, max=1.0)

    bpy.types.Scene.xlr_filename_template = StringProperty(
        name="Filename Template",
        default="{cam_coll}/{env}_{cam}_{frame:04d}",
        description="Fields: cam, frame, cam_tag, marker, cam_coll, env, env_tag.",
    )
    bpy.types.Scene.xlr_caption_template = StringProperty(
        name="Caption Template",
        default="{cam_tag}",
        description="Fields: cam, frame, cam_tag, marker, cam_coll, env, env_tag.",
    )

    bpy.types.Scene.xlr_camera_collections = CollectionProperty(type=XLR_CameraCollectionItem)
    bpy.types.Scene.xlr_camera_collections_index = IntProperty(default=0)

    bpy.types.Scene.xlr_envs = CollectionProperty(type=XLR_EnvItem)
    bpy.types.Scene.xlr_envs_index = IntProperty(default=0)

    bpy.types.Scene.xlr_world_bake_resolution = IntProperty(
        name="World Bake Resolution",
        default=2048,
        min=256,
        max=16384,
    )

    bpy.types.Scene.xlr_world_bake_samples = IntProperty(
        name="World Bake Samples",
        default=4,
        min=1,
        max=1024,
    )

    bpy.types.Scene.camera_rename_prefix = StringProperty(
        name="Camera Rename Prefix",
        default="{cam_coll}_",
        description="Fields: cam, cam_tag, cam_coll, index.",
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
    del bpy.types.Scene.xlr_envs
    del bpy.types.Scene.xlr_envs_index
    del bpy.types.Scene.xlr_world_bake_resolution
    del bpy.types.Scene.xlr_world_bake_samples
    del bpy.types.Scene.camera_rename_prefix
    del bpy.types.Object.xlr_tag


if __name__ == "__main__":
    register()
