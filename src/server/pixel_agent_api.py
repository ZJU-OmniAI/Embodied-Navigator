#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI service for deploying DThinkVLN pixel-action agents on a real robot.

API design (minimal):
- POST /reset : reset agent state + set a new instruction
- POST /step  : upload observation + poses, return MOVE_TO pixel action or STOP

This mirrors the online loop in `src/eval/evaluate_multi.py`, but exposes the
agent over HTTP so the robot stack can call it step-by-step.

Notes
-----
- This repo environment may not include FastAPI/uvicorn by default.
  Install (example):
    pip install "fastapi>=0.110" "uvicorn[standard]"
- Observations accept multi-view RGB images (rgb_front/left/right/back).
  Each image can be:
    - {"b64": "<base64>", "encoding": "jpg|png", "colorspace": "RGB|BGR"}
    - data URI string: "data:image/jpeg;base64,..."
    - raw base64 string (jpg/png bytes)
    - nested list (H x W x C) of ints
"""

from __future__ import annotations

import base64
import io
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "FastAPI is required to run this service. Install with: "
        'pip install "fastapi>=0.110" "uvicorn[standard]"'
    ) from e

from ..utils import agent_registry
from ..agent import *  # noqa: F401,F403 - ensure agents are registered


def _now_ms() -> int:
    return int(time.time() * 1000)


_TARGET_IMAGE_SIZE = 280  # crop+resize to 280x280 on the server side


def _strip_data_uri(s: str) -> str:
    # data:image/jpeg;base64,xxxx
    if s.startswith("data:") and "base64," in s:
        return s.split("base64,", 1)[1]
    return s


def _center_crop_square_rgb(arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Center-crop an RGB image to a square; returns (cropped, meta)."""
    h, w = int(arr.shape[0]), int(arr.shape[1])
    size = min(h, w)
    top = int((h - size) // 2)
    left = int((w - size) // 2)
    cropped = arr[top: top + size, left: left + size]
    meta = {"orig_h": h, "orig_w": w, "crop_top": top,
            "crop_left": left, "crop_size": int(size)}
    return cropped, meta


def _resize_rgb(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    im = Image.fromarray(arr, mode="RGB")
    im = im.resize((int(size), int(size)), resample=Image.BILINEAR)
    return np.asarray(im)


def _decode_image_to_pil_rgb(value: Any) -> Optional[Image.Image]:
    """
    Best-effort decode for an image payload into a PIL RGB image.
    Returns None if decoding fails.
    """
    if value is None:
        return None

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = arr[..., :3]
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        return None

    if isinstance(value, dict):
        b64 = value.get("b64") or value.get("base64")
        if isinstance(b64, str) and b64.strip():
            colorspace = str(value.get("colorspace") or "RGB").upper()
            try:
                raw = base64.b64decode(_strip_data_uri(b64))
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                if colorspace == "BGR":
                    arr = np.asarray(
                        img)[..., ::-1].astype(np.uint8, copy=False)
                    return Image.fromarray(arr, mode="RGB")
                return img
            except Exception:
                return None
        return None

    if isinstance(value, str) and value.strip():
        try:
            raw = base64.b64decode(_strip_data_uri(value))
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    if isinstance(value, list):
        try:
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = arr[..., :3]
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                return Image.fromarray(arr, mode="RGB")
            return None
        except Exception:
            return None

    return None


def _get_obs_image_payload(obs: Dict[str, Any], sensor: str) -> Any:
    """Fetch a per-sensor image payload from obs, supporting both obs['images'][sensor] and obs[sensor]."""
    if not isinstance(obs, dict) or not isinstance(sensor, str) or not sensor:
        return None
    images = obs.get("images")
    if isinstance(images, dict) and sensor in images:
        return images.get(sensor)
    return obs.get(sensor)


def _decode_image_and_preprocess(value: Any, *, target_size: int) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Decode an image payload into a numpy array (RGB) for the agent.
    Then center-crop and resize to target_size x target_size.

    Returns (processed_value, transform_meta). If decoding fails, returns (original, None).
    """
    if value is None:
        return None, None

    # Already a numpy array (e.g. caller is python-internal)
    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            rgb = arr[..., :3]
            cropped, meta = _center_crop_square_rgb(rgb)
            resized = _resize_rgb(cropped, target_size)
            meta["resized_h"] = int(target_size)
            meta["resized_w"] = int(target_size)
            return resized, meta
        return value, None

    # Pydantic may parse dict payloads
    if isinstance(value, dict):
        b64 = value.get("b64") or value.get("base64")
        if isinstance(b64, str) and b64.strip():
            colorspace = str(value.get("colorspace") or "RGB").upper()
            try:
                raw = base64.b64decode(_strip_data_uri(b64))
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                arr = np.asarray(img)  # RGB
                if colorspace == "BGR":
                    arr = arr[..., ::-1]
                cropped, meta = _center_crop_square_rgb(arr)
                resized = _resize_rgb(cropped, target_size)
                meta["resized_h"] = int(target_size)
                meta["resized_w"] = int(target_size)
                return resized, meta
            except Exception:
                return value, None
        return value, None

    # data URI / raw base64 string
    if isinstance(value, str) and value.strip():
        try:
            raw = base64.b64decode(_strip_data_uri(value))
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            arr = np.asarray(img)
            cropped, meta = _center_crop_square_rgb(arr)
            resized = _resize_rgb(cropped, target_size)
            meta["resized_h"] = int(target_size)
            meta["resized_w"] = int(target_size)
            return resized, meta
        except Exception:
            return value, None

    # Nested list -> np.array
    if isinstance(value, list):
        try:
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                rgb = arr[..., :3].astype(np.uint8, copy=False)
                cropped, meta = _center_crop_square_rgb(rgb)
                resized = _resize_rgb(cropped, target_size)
                meta["resized_h"] = int(target_size)
                meta["resized_w"] = int(target_size)
                return resized, meta
            return arr, None
        except Exception:
            return value, None

    return value, None


def _decode_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize observation schema to what DThinkPixelAgent expects.
    - Supports obs["images"] mapping or direct rgb_* keys.
    - Decodes images into numpy arrays where possible.
    - Center-crops and resizes each rgb_* image to 280x280.

    The per-sensor crop/resize metadata is stored in `obs["image_transforms"]` and is used
    to invert predicted pixels back to original coordinates on response.
    """
    if not isinstance(obs, dict):
        return {}

    out = dict(obs)
    transforms: Dict[str, Any] = {}
    if "images" in out and isinstance(out["images"], dict):
        images = {}
        for k, v in out["images"].items():
            vv, meta = _decode_image_and_preprocess(
                v, target_size=_TARGET_IMAGE_SIZE)
            images[k] = vv
            if meta is not None:
                transforms[k] = meta
        out["images"] = images
        # Provide convenient top-level keys (agent supports both).
        for k, v in images.items():
            out.setdefault(k, v)
        if transforms:
            out["image_transforms"] = transforms
        return out

    # Otherwise decode top-level rgb_* keys
    for k in list(out.keys()):
        if isinstance(k, str) and k.startswith("rgb_"):
            vv, meta = _decode_image_and_preprocess(
                out[k], target_size=_TARGET_IMAGE_SIZE)
            out[k] = vv
            if meta is not None:
                transforms[k] = meta
    if transforms:
        out["image_transforms"] = transforms
    return out


def _invert_pixel_from_280(pixel: Any, meta: Dict[str, Any]) -> Any:
    """Invert pixel from 280x280 back to original image space using meta."""
    if not (isinstance(pixel, (list, tuple)) and len(pixel) >= 2):
        return pixel
    try:
        u = float(pixel[0])
        v = float(pixel[1])
    except (TypeError, ValueError):
        return pixel

    resized = float(meta.get("resized_w") or _TARGET_IMAGE_SIZE)
    crop_left = float(meta.get("crop_left") or 0.0)
    crop_top = float(meta.get("crop_top") or 0.0)
    crop_size = float(meta.get("crop_size") or 0.0)
    if crop_size <= 0.0 or resized <= 0.0:
        return pixel

    # Clamp to resized image bounds for safety.
    u = max(0.0, min(resized - 1.0, u))
    v = max(0.0, min(resized - 1.0, v))

    scale = crop_size / resized
    return [crop_left + u * scale, crop_top + v * scale]


def _normalize_action_for_env(action: Any, default_sensor: Optional[str]) -> Dict[str, Any]:
    """
    Keep the same normalization behavior as `src/eval/evaluate_multi.py` so the
    caller (robot/env) can directly consume the result if desired.
    """
    if not isinstance(action, dict):
        return {"type": "CMD", "action": "STOP"}

    out = dict(action)
    out["type"] = out.get("type") or out.get("action_type") or "CMD"

    if out["type"] == "MOVE_TO":
        pixel = out.get("pixel") or out.get("action")
        if not (isinstance(pixel, (list, tuple)) and len(pixel) >= 2):
            return {"type": "CMD", "action": "STOP"}
        out["action"] = [float(pixel[0]), float(pixel[1])]

        sensor = out.get("sensor") or out.get("choice") or default_sensor or ""
        if sensor.startswith("rgb"):
            sensor = sensor.replace("rgb", "depth", 1)
        out["sensor"] = sensor or default_sensor or "depth_front"
    else:
        out["action"] = out.get("action") or "STOP"

    return out


def _to_serializable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    return x


class ResetRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="If omitted, a new session_id is created.")
    instruction: str
    episode_id: Optional[str] = None
    first_obs: Optional[Dict[str, Any]] = None
    sensor_poses: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None


class ResetResponse(BaseModel):
    session_id: str
    step_id: int
    episode_id: Optional[str] = None
    ts_ms: int


class StepRequest(BaseModel):
    session_id: str
    obs: Dict[str, Any]
    sensor_poses: Optional[Dict[str, Any]] = None
    return_model_text: bool = False
    return_model_out: bool = False
    return_depth: bool = False
    depth_sensor: Optional[str] = Field(
        default=None,
        description="Which RGB sensor key to run depth estimation on (e.g. 'rgb_front'). "
        "If omitted, uses the model-selected view when available.",
    )
    depth_camera: Optional[str] = Field(
        default=None,
        description="Camera name for intrinsics lookup in /cam_info. Defaults to depth_sensor if omitted.",
    )


class StepResponse(BaseModel):
    session_id: str
    step_id: int
    action: Dict[str, Any]
    env_action: Dict[str, Any]
    model_text: Optional[str] = None
    model_out: Optional[Dict[str, Any]] = None
    depth: Optional[Dict[str, Any]] = None
    ts_ms: int


class CamInfoUpsertRequest(BaseModel):
    camera: str = Field(..., description="Camera name used as lookup key.")
    intrinsics: Any = Field(...,
                            description="Camera intrinsics matrix K (3x3).")
    hw: Optional[Tuple[int, int]] = Field(
        default=None,
        description="(H, W) resolution that the intrinsics correspond to. If provided and input image differs, K is scaled.",
    )


class CamInfoResponse(BaseModel):
    camera: str
    intrinsics: Any
    hw: Optional[Tuple[int, int]] = None
    ts_ms: int


class Da3DepthRequest(BaseModel):
    image: Any = Field(..., description="Image payload (same formats as obs images: base64/dataURI/list/ndarray-like).")
    camera: Optional[str] = Field(
        default=None,
        description="If set, lookup intrinsics from /cam_info using this camera name (unless intrinsics is provided).",
    )
    intrinsics: Optional[Any] = Field(
        default=None,
        description="Optional intrinsics matrix K (3x3). If provided, overrides camera lookup.",
    )
    hw: Optional[Tuple[int, int]] = Field(
        default=None,
        description="(H, W) resolution that provided intrinsics correspond to. If input image differs, K is scaled.",
    )
    process_res: int = Field(
        default=640, description="DA3 processing resolution.")
    process_res_method: str = Field(
        default="upper_bound_resize", description="DA3 processing resize method.")


class Da3DepthResponse(BaseModel):
    depth: Dict[str, Any]
    ts_ms: int


class _NullPixelAgent:
    """Fallback agent when no model is configured; always returns STOP."""

    def __init__(self) -> None:
        # DThinkAgent can be constructed without a model; it gives us a valid AgentState container.
        self._state_agent = agent_registry.create("dthink_agent")

    def reset(self, instruction: str, first_obs=None, *, sensor_poses=None, episode_id=None, meta=None):
        # Keep shape compatible with AgentState (we don't require its type here).
        return self._state_agent.reset(
            instruction=instruction,
            first_obs=first_obs,
            sensor_poses=sensor_poses,
            episode_id=episode_id,
            meta=meta,
        )

    def step(self, obs, state, *, sensor_poses=None):
        action = {"type": "CMD", "action": "STOP", "choice": ""}
        model_out = {"text": "", "action": action}
        return action, state, model_out


def create_app(
    *,
    agent_type: str = "dthink_pixel_agent",
    model_path: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> FastAPI:
    """
    Factory so you can run with uvicorn:
      uvicorn src.serve.pixel_agent_api:app --host 0.0.0.0 --port 8000
    """

    app = FastAPI(title="DThinkVLN Pixel Agent Service")

    # Shared model instance; sessions only store AgentState.
    # If model_path is not configured, keep the service alive with a STOP-only agent.
    if model_path:
        agent = agent_registry.create(
            agent_type,
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        agent = _NullPixelAgent()
    agent_lock = threading.Lock()

    sessions: Dict[str, Any] = {}
    cam_db: Dict[str, Dict[str, Any]] = {}

    da3_lock = threading.Lock()
    da3_model = None
    da3_device = None
    da3_load_error: Optional[str] = None
    da3_model_path = _get_env(
        "DTHINK_DA3_MODEL_PATH", "model/share/DA3NESTED-GIANT-LARGE-1.1") or "model/share/DA3NESTED-GIANT-LARGE-1.1"

    def _parse_bool_env(name: str, default: str) -> bool:
        v = (_get_env(name, default) or default).strip().lower()
        return v not in ("0", "false", "no", "off", "")

    def _load_da3_model_once() -> None:
        """
        Load DA3 at server startup (not on first request).
        We intentionally do not fall back to lazy loading inside handlers.
        """
        nonlocal da3_model, da3_device, da3_load_error
        if da3_model is not None:
            return
        if da3_load_error is not None:
            raise RuntimeError(da3_load_error)

        try:
            try:
                from depth_anything_3.api import DepthAnything3  # type: ignore
                import torch  # type: ignore
            except ModuleNotFoundError:
                # Fallback to the cached source tree if available.
                proj_root = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), "..", "..", ".."))
                da3_src = os.path.join(proj_root, "cache", "Depth-Anything-3", "src")
                if os.path.isdir(da3_src) and da3_src not in sys.path:
                    sys.path.insert(0, da3_src)
                from depth_anything_3.api import DepthAnything3  # type: ignore
                import torch  # type: ignore

            da3_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            da3_model = DepthAnything3.from_pretrained(da3_model_path)
            da3_model = da3_model.to(da3_device)
            da3_model.eval()
        except Exception as e:
            da3_load_error = f"DA3 load failed: {type(e).__name__}: {e}"
            raise

    def _require_da3_loaded() -> None:
        if da3_model is not None:
            return
        detail = da3_load_error or "DA3 model is not loaded."
        raise HTTPException(status_code=503, detail=detail)

    if _parse_bool_env("DTHINK_PRELOAD_DA3", "1"):
        @app.on_event("startup")
        def _startup_preload_da3() -> None:
            # Load once before serving requests.
            with da3_lock:
                _load_da3_model_once()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "agent_type": agent_type,
            "has_model_path": bool(model_path),
            "sessions": len(sessions),
            "cams": len(cam_db),
            "da3_loaded": bool(da3_model is not None),
            "da3_model_path": da3_model_path,
            "ts_ms": _now_ms(),
        }

    @app.post("/cam_info", response_model=CamInfoResponse)
    def cam_info(req: CamInfoUpsertRequest) -> CamInfoResponse:
        camera = (req.camera or "").strip()
        if camera == "":
            raise HTTPException(
                status_code=400, detail="camera must be a non-empty string")

        try:
            k = np.asarray(req.intrinsics, dtype=np.float32)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"invalid intrinsics: {e}") from e
        if k.shape != (3, 3):
            raise HTTPException(
                status_code=400, detail=f"intrinsics must have shape (3,3), got {tuple(k.shape)}")

        hw = None
        if req.hw is not None:
            try:
                h = int(req.hw[0])
                w = int(req.hw[1])
                if h <= 0 or w <= 0:
                    raise ValueError("hw must be positive")
                hw = (h, w)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"invalid hw: {e}") from e

        cam_db[camera] = {"intrinsics": k, "hw": hw, "ts_ms": _now_ms()}
        return CamInfoResponse(camera=camera, intrinsics=k.tolist(), hw=hw, ts_ms=cam_db[camera]["ts_ms"])

    def _resolve_intrinsics_for_da3(
        *,
        camera: Optional[str],
        intrinsics: Optional[Any],
        hw: Optional[Tuple[int, int]],
        image_hw: Tuple[int, int],
    ) -> Tuple[str, np.ndarray]:
        """
        Resolve intrinsics for DA3:
        - If `intrinsics` is provided: use it (optionally scaling using req.hw).
        - Else lookup from cam_db using `camera` (and scale using stored hw if present).

        Returns (camera_key, K_3x3).
        """
        cam_key = (camera or "").strip()
        k = None
        k_hw = None

        if intrinsics is not None:
            try:
                k = np.asarray(intrinsics, dtype=np.float32)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"invalid intrinsics: {e}") from e
            if k.shape != (3, 3):
                raise HTTPException(
                    status_code=400, detail=f"intrinsics must have shape (3,3), got {tuple(k.shape)}")
            k_hw = hw
            cam_key = cam_key or "inline"
        else:
            if cam_key == "":
                raise HTTPException(
                    status_code=400, detail="must provide either intrinsics or camera")

            candidates = [cam_key]
            if cam_key.startswith("rgb_"):
                candidates.append(cam_key[len("rgb_"):])
            if cam_key.startswith("rgb"):
                candidates.append(cam_key.replace("rgb", "", 1).lstrip("_"))

            cam_rec = None
            found_key = None
            for c in candidates:
                if c in cam_db:
                    found_key = c
                    cam_rec = cam_db[c]
                    break
            if cam_rec is None:
                raise HTTPException(
                    status_code=400, detail=f"no /cam_info found for {candidates}")
            cam_key = found_key or cam_key
            k = np.asarray(cam_rec["intrinsics"], dtype=np.float32)
            k_hw = cam_rec.get("hw")

        # Optional scaling if intrinsics hw is known and differs from image hw.
        h1, w1 = int(image_hw[0]), int(image_hw[1])
        if isinstance(k_hw, (tuple, list)) and len(k_hw) == 2:
            try:
                h0, w0 = int(k_hw[0]), int(k_hw[1])
            except Exception:
                h0, w0 = 0, 0
            if h0 > 0 and w0 > 0 and (h1 != h0 or w1 != w0):
                sx = float(w1) / float(w0)
                sy = float(h1) / float(h0)
                k = k.copy()
                k[0, 0] *= sx
                k[1, 1] *= sy
                k[0, 2] *= sx
                k[1, 2] *= sy

        return cam_key, k

    @app.post("/da3_depth", response_model=Da3DepthResponse)
    @app.post("/depth", response_model=Da3DepthResponse)
    def da3_depth(req: Da3DepthRequest) -> Da3DepthResponse:
        pil = _decode_image_to_pil_rgb(req.image)
        if pil is None:
            raise HTTPException(status_code=400, detail="cannot decode image")

        # PIL: (W, H)
        image_hw = (int(pil.size[1]), int(pil.size[0]))
        cam_key, k = _resolve_intrinsics_for_da3(
            camera=req.camera,
            intrinsics=req.intrinsics,
            hw=req.hw,
            image_hw=image_hw,
        )

        with da3_lock:
            # Serialize DA3 inference to avoid GPU contention and keep model init singletons safe.
            nonlocal da3_model, da3_device
            _require_da3_loaded()

            pred = da3_model.inference(
                image=[pil],
                intrinsics=np.expand_dims(k, 0),
                process_res=int(req.process_res),
                process_res_method=str(req.process_res_method),
                export_dir=None,
                export_format="mini_npz",
            )

        depth = np.asarray(pred.depth[0], dtype=np.float32)  # (H, W)
        depth_b64 = base64.b64encode(depth.tobytes()).decode("ascii")
        depth_payload = {
            "camera": cam_key,
            "hw": [int(depth.shape[0]), int(depth.shape[1])],
            "dtype": "float32",
            "encoding": "raw_f32_le_b64",
            "b64": depth_b64,
            "min": float(np.min(depth)) if depth.size else 0.0,
            "max": float(np.max(depth)) if depth.size else 0.0,
            "intrinsics_used": k.tolist(),
            "image_hw": [int(image_hw[0]), int(image_hw[1])],
        }
        return Da3DepthResponse(depth=_to_serializable(depth_payload), ts_ms=_now_ms())

    @app.post("/reset", response_model=ResetResponse)
    def reset(req: ResetRequest) -> ResetResponse:
        session_id = req.session_id or str(uuid.uuid4())
        first_obs = _decode_observation(
            req.first_obs) if req.first_obs is not None else None
        sensor_poses = req.sensor_poses

        with agent_lock:
            state = agent.reset(
                instruction=req.instruction,
                first_obs=first_obs,
                sensor_poses=sensor_poses,
                episode_id=req.episode_id,
                meta=req.meta,
            )
        sessions[session_id] = state
        return ResetResponse(
            session_id=session_id,
            step_id=getattr(state, "step_id", 0),
            episode_id=getattr(state, "episode_id", None),
            ts_ms=_now_ms(),
        )

    @app.post("/step", response_model=StepResponse)
    def step(req: StepRequest) -> StepResponse:
        if req.session_id not in sessions:
            raise HTTPException(
                status_code=404, detail=f"Unknown session_id: {req.session_id}")

        state = sessions[req.session_id]
        obs = _decode_observation(req.obs)
        sensor_poses = req.sensor_poses

        default_sensor = "depth_front"
        if isinstance(obs, dict):
            # Try to infer default from available RGB views.
            for s in ("rgb_front", "rgb_left", "rgb_right", "rgb_back"):
                if s in obs or (isinstance(obs.get("images"), dict) and s in obs["images"]):
                    default_sensor = s.replace("rgb", "depth", 1)
                    break
        with agent_lock:
            action, new_state, model_out = agent.step(
                obs, state, sensor_poses=sensor_poses)
        sessions[req.session_id] = new_state

        # If the server resized images to 280x280, the model predicts pixels in that space.
        # Convert them back to original image coordinates using the stored transform.
        if isinstance(action, dict) and action.get("type") == "MOVE_TO":
            sensor = action.get("choice") or action.get("sensor") or ""
            transforms = obs.get("image_transforms") if isinstance(
                obs, dict) else None
            meta = transforms.get(sensor) if isinstance(
                transforms, dict) and isinstance(sensor, str) else None
            if isinstance(meta, dict):
                pix = action.get("pixel") or action.get("action")
                pix0 = _invert_pixel_from_280(pix, meta)
                action = dict(action)
                action["pixel"] = pix0
                action["action"] = pix0
                if isinstance(model_out, dict):
                    model_out = dict(model_out)
                    model_out["action"] = action

        env_action = _normalize_action_for_env(action, default_sensor)
        model_text = None
        if req.return_model_text and isinstance(model_out, dict):
            model_text = model_out.get("text", "")
        print(model_text)
        depth_payload = None
        if req.return_depth:
            # Choose the sensor for depth: explicit request -> model selected view -> fallback.
            depth_sensor = (req.depth_sensor or "").strip()
            if depth_sensor == "":
                if isinstance(action, dict):
                    depth_sensor = str(action.get("choice")
                                       or action.get("sensor") or "").strip()
            if depth_sensor == "":
                # fallback to common key
                depth_sensor = "rgb_front"

            # Pick camera name for intrinsics lookup.
            cam_name = (req.depth_camera or depth_sensor).strip()
            candidates = [cam_name]
            if cam_name.startswith("rgb_"):
                candidates.append(cam_name[len("rgb_"):])
            if cam_name.startswith("rgb"):
                candidates.append(cam_name.replace("rgb", "", 1).lstrip("_"))
            cam_rec = None
            cam_key = None
            for c in candidates:
                if c in cam_db:
                    cam_key = c
                    cam_rec = cam_db[c]
                    break
            if cam_rec is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"return_depth requires intrinsics; no /cam_info found for {candidates}",
                )

            img_payload = _get_obs_image_payload(req.obs, depth_sensor)
            pil = _decode_image_to_pil_rgb(img_payload)
            if pil is None:
                raise HTTPException(
                    status_code=400, detail=f"cannot decode image for sensor '{depth_sensor}'")

            k = np.asarray(cam_rec["intrinsics"], dtype=np.float32)
            hw0 = cam_rec.get("hw")
            if isinstance(hw0, (tuple, list)) and len(hw0) == 2:
                h0, w0 = int(hw0[0]), int(hw0[1])
                w1, h1 = pil.size  # PIL: (W, H)
                if h0 > 0 and w0 > 0 and (h1 != h0 or w1 != w0):
                    sx = float(w1) / float(w0)
                    sy = float(h1) / float(h0)
                    k = k.copy()
                    k[0, 0] *= sx
                    k[1, 1] *= sy
                    k[0, 2] *= sx
                    k[1, 2] *= sy

            with da3_lock:
                # Serialize DA3 inference to avoid GPU contention and keep model init singletons safe.
                nonlocal da3_model, da3_device
                _require_da3_loaded()

                pred = da3_model.inference(
                    image=[pil],
                    intrinsics=np.expand_dims(k, 0),
                    export_dir=None,
                    export_format="mini_npz",
                )
            depth = np.asarray(pred.depth[0], dtype=np.float32)  # (H, W)
            depth_b64 = base64.b64encode(depth.tobytes()).decode("ascii")
            depth_payload = {
                "sensor": depth_sensor,
                "camera": cam_key,
                "hw": [int(depth.shape[0]), int(depth.shape[1])],
                "dtype": "float32",
                "encoding": "raw_f32_le_b64",
                "b64": depth_b64,
                "min": float(np.min(depth)) if depth.size else 0.0,
                "max": float(np.max(depth)) if depth.size else 0.0,
                "intrinsics_used": k.tolist(),
                "image_hw": [int(pil.size[1]), int(pil.size[0])],
            }

        out = StepResponse(
            session_id=req.session_id,
            step_id=getattr(new_state, "step_id", 0),
            action=_to_serializable(action),
            env_action=_to_serializable(env_action),
            model_text=model_text,
            model_out=_to_serializable(
                model_out) if req.return_model_out else None,
            depth=_to_serializable(depth_payload),
            ts_ms=_now_ms(),
        )
        return out

    return app


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip()
    return v if v != "" else default


# Default module-level app for uvicorn discovery.
app = create_app(
    agent_type=_get_env("DTHINK_AGENT_TYPE",
                        "dthink_pixel_agent") or "dthink_pixel_agent",
    model_path=_get_env("DTHINK_MODEL_PATH"),
    max_new_tokens=int(_get_env("DTHINK_MAX_NEW_TOKENS", "512") or "512"),
    temperature=float(_get_env("DTHINK_TEMPERATURE", "0.2") or "0.2"),
    top_p=float(_get_env("DTHINK_TOP_P", "0.9") or "0.9"),
)
