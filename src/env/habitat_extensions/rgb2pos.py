from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class SensorGeometry:
    """
    - 静态函数：仅依赖传入数据（cfg / agent_state / 像素 / 深度 / 外参与内参）
    - 动态接口：从 env 取 K 与 T_world_cam，再转调静态函数

    约定（Habitat 摄像机）：
      相机系： right=+X, up=+Y, forward=-Z
      像素系： 左上为原点，u→右，v→下
    """

    # =========================
    # 静态：基础变换与内参
    # =========================
    @staticmethod
    def quat_wxyz_to_R(q_wxyz: np.ndarray) -> np.ndarray:
        q = np.asarray(q_wxyz, dtype=np.float32).reshape(4)
        w, x, y, z = q
        R = np.array([
            [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
            [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
            [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)],
        ], dtype=np.float32)
        return R

    @staticmethod
    def build_T_world_X(position_xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        p = np.asarray(position_xyz, dtype=np.float32).reshape(3)
        R = SensorGeometry.quat_wxyz_to_R(quat_wxyz)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    @staticmethod
    def invert_T(T: np.ndarray) -> np.ndarray:
        R = T[:3, :3]
        t = T[:3, 3]
        Ti = np.eye(4, dtype=np.float32)
        Ri = R.T
        Ti[:3, :3] = Ri
        Ti[:3, 3] = -Ri @ t
        return Ti

    @staticmethod
    def intrinsics_from_fov(
        width: int, height: int, hfov_deg: float, vfov_deg: Optional[float] = None
    ) -> Dict[str, Any]:
        W, H = int(width), int(height)
        assert W > 0 and H > 0 and hfov_deg > 0, "width/height/hfov 不合法"
        hfov = np.deg2rad(float(hfov_deg))
        fx = W / (2.0 * np.tan(hfov / 2.0))
        if vfov_deg is not None and vfov_deg > 0:
            vfov = np.deg2rad(float(vfov_deg))
            fy = H / (2.0 * np.tan(vfov / 2.0))
        else:
            fy = fx * (W / H)
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        return {"width": W, "height": H, "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy), "K": K}

    # =========================
    # 🔹静态：从 cfg / agent_state 提取 K 与外参
    # =========================
    @staticmethod
    def _static_resolve_sensor_name(sensor_names: List[str], sensor_name: Optional[str]) -> str:
        if not sensor_names:
            raise RuntimeError("当前 Agent 无任何传感器")
        if sensor_name is None:
            return sensor_names[0]
        if sensor_name in sensor_names:
            return sensor_name
        if sensor_name.endswith("_sensor") and sensor_name[:-7] in sensor_names:
            return sensor_name[:-7]
        if (not sensor_name.endswith("_sensor")) and sensor_name + "_sensor" in sensor_names:
            return sensor_name + "_sensor"
        raise KeyError(f"未找到传感器 '{sensor_name}'，可用：{sensor_names}")

    @staticmethod
    def get_T_world_agent_static(agent_state) -> np.ndarray:
        """从 sim.get_agent_state() 的返回对象构造 T_world_agent（静态）"""
        q = np.array([agent_state.rotation.w, agent_state.rotation.x,
                      agent_state.rotation.y, agent_state.rotation.z], dtype=np.float32)
        p = np.array(agent_state.position, dtype=np.float32)
        return SensorGeometry.build_T_world_X(p, q)

    @staticmethod
    def get_T_world_sensor_static(agent_state, sensor_name: Optional[str] = None) -> np.ndarray:
        """
        从 agent_state（包含 sensor_states 字典）获取 T_world_sensor（静态）
        """
        sensor_names = list(agent_state.sensor_states.keys())
        name = SensorGeometry._static_resolve_sensor_name(
            sensor_names, sensor_name)
        sst = agent_state.sensor_states[name]
        q = np.array([sst.rotation.w, sst.rotation.x,
                     sst.rotation.y, sst.rotation.z], dtype=np.float32)
        p = np.array(sst.position, dtype=np.float32)
        return SensorGeometry.build_T_world_X(p, q)

    @staticmethod
    def get_K_from_cfg_static(cfg, sensor_name: Optional[str] = None, agent_name: str = "main_agent") -> Dict[str, Any]:
        """
        从 Habitat cfg（OmegaConf）提取指定传感器的 width/height/hfov(/vfov) 并构造 K（静态）
        """
        sensors = cfg.habitat.simulator.agents.__getattr__(
            agent_name).sim_sensors

        # 先枚举可用 uuid（尽量从配置里推断）
        all_names = []
        for k in sensors:
            c = getattr(sensors, k)
            uid = getattr(c, "uuid", None)
            if isinstance(uid, str):
                all_names.append(uid)
            else:
                all_names.append(k)

        # 解析目标名
        name = SensorGeometry._static_resolve_sensor_name(
            all_names, sensor_name)

        # 在配置中定位该 uuid 对应节点
        if hasattr(sensors, f"{name}_sensor"):
            scfg = getattr(sensors, f"{name}_sensor")
        elif hasattr(sensors, name):
            scfg = getattr(sensors, name)
        else:
            scfg = None
            for k in sensors:
                c = getattr(sensors, k)
                if getattr(c, "uuid", None) == name:
                    scfg = c
                    break
            if scfg is None:
                raise KeyError(f"cfg 中未找到与 uuid='{name}' 对应的传感器配置")

        W = int(getattr(scfg, "width", 0))
        H = int(getattr(scfg, "height", 0))
        hfov_deg = float(getattr(scfg, "hfov", 0.0))
        vfov_deg = getattr(scfg, "vfov", None)
        vfov_deg = float(vfov_deg) if vfov_deg is not None else None

        if W <= 0 or H <= 0 or hfov_deg <= 0:
            raise ValueError(
                f"width/height/hfov 无效: W={W}, H={H}, hfov={hfov_deg}")

        return SensorGeometry.intrinsics_from_fov(W, H, hfov_deg, vfov_deg)

    @staticmethod
    def _get_depth_cfg(cfg, sensor_name: Optional[str] = None, agent_name: str = "main_agent"):
        """
        从 cfg 中拿到指定深度传感器的配置结点（含 min_depth / max_depth / 可能的 normalize_depth）。
        兼容 uuid == 'depth' / 'depth_sensor' 等写法。
        """
        sensors = cfg.habitat.simulator.agents.__getattr__(
            agent_name).sim_sensors
        sensor_names = list(sensors.keys())
        name = SensorGeometry._static_resolve_sensor_name(
            sensor_names, sensor_name)
        return getattr(sensors, name)

    @staticmethod
    def _depth_raw_to_meters(d_raw: float, scfg) -> float:
        """
        把原始深度值转换成米。若配置存在 normalize_depth 且为 True，则按 [min,max] 反归一化。
        """
        min_d = float(getattr(scfg, "min_depth", 0.0))
        max_d = float(getattr(scfg, "max_depth", 10.0))
        norm_flag = getattr(scfg, "normalize_depth", None)  # 有的版本没有这个字段

        if norm_flag is True:
            return d_raw * (max_d - min_d) + min_d
        return float(d_raw)

    # ===== 静态：世界点 -> 相机系点 =====
    @staticmethod
    def world_to_cam_point(p_world: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
        """
        世界点 -> 相机坐标系点（3,）
        T_world_cam: 相机外参（世界到相机：不是！本函数要求 T_world_cam = [R_wc|t_wc] 表示“相机在世界中的位姿”）
        这里用 p_cam = R_wc^T * (p_world - t_wc)
        """
        p_world = np.asarray(p_world, dtype=np.float32).reshape(3)
        R_wc = T_world_cam[:3, :3]
        t_wc = T_world_cam[:3, 3]
        return R_wc.T @ (p_world - t_wc)

    # ===== 静态：相机系点 -> 像素 =====
    @staticmethod
    def cam_point_to_pixel(
        p_cam: np.ndarray, K: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        相机坐标点 -> 像素坐标 (u,v)，并返回两种深度：
          - z_depth: 相机系 Z 分量的“前向距离” (正值)，即 z_depth = -z_cam
          - ray_depth: 从光心到点的欧氏距离
        约定：forward = -Z，所以可见点满足 z_cam < 0
        像素投影：
          x' = x_cam / (-z_cam)
          y' = y_cam / (-z_cam)
          u  = fx * x' + cx
          v  = -fy * y' + cy     （像素 v 向下为正）
        """
        x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # 可见性：在相机前方（z_cam < 0）
        if z >= 0.0:
            # 返回 NaN 以示不可见；由上层决定如何处理
            return (np.nan, np.nan, np.nan, np.nan)

        inv_mz = 1.0 / (-z)  # -z_cam > 0
        u = fx * (x * inv_mz) + cx
        v = -fy * (y * inv_mz) + cy  # 注意像素 v 向下

        z_depth = -z                    # 前向距离（projective depth）
        ray_depth = float(np.linalg.norm([x, y, z], ord=2))  # 欧氏距离
        return (u, v, z_depth, ray_depth)

    # ===== 静态：世界点 -> 像素（整合） =====
    @staticmethod
    def world_to_pixel_static(
        p_world: np.ndarray,
        K: np.ndarray,
        T_world_cam: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,  # (W, H)
        clip_outside: bool = True,
        return_depth: bool = True,
    ) -> Dict[str, Any]:
        """
        纯函数：世界点 -> 像素
        返回：
          {
            "u": float or nan,
            "v": float or nan,
            "visible": bool,          # 在相机前方且(可选)像素落在图像内
            "z_depth": float or nan,  # -z_cam（前向距离，单位：米）
            "ray_depth": float or nan # 欧氏距离（单位：米）
          }
        """
        p_cam = SensorGeometry.world_to_cam_point(p_world, T_world_cam)
        u, v, z_depth, ray_depth = SensorGeometry.cam_point_to_pixel(p_cam, K)

        # 是否在相机前方
        visible = not (np.isnan(u) or np.isnan(v))

        # 可选：裁剪到图像范围
        if visible and image_size is not None and clip_outside:
            W, H = int(image_size[0]), int(image_size[1])
            if not (0.0 <= u < W and 0.0 <= v < H):
                visible = False

        out = {"u": u, "v": v, "visible": bool(visible)}
        if return_depth:
            out.update({"z_depth": z_depth, "ray_depth": ray_depth})
        return out

    # ===== 静态：批量版本 =====
    @staticmethod
    def world_points_to_pixels_batch_static(
        P_world: np.ndarray,
        K: np.ndarray,
        T_world_cam: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
        clip_outside: bool = True,
        return_depth: bool = True,
    ) -> Dict[str, Any]:
        """
        批量世界点 -> 像素；P_world: (N,3)
        返回 dict，包含 u/v/visible/(z_depth)/(ray_depth)，每项 shape (N,)
        """
        P_world = np.asarray(P_world, dtype=np.float32).reshape(-1, 3)
        R_wc = T_world_cam[:3, :3]
        t_wc = T_world_cam[:3, 3]
        # p_cam = R^T (p_w - t)
        P_cam = (R_wc.T @ (P_world - t_wc[None, :]).T).T  # (N,3)

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        x, y, z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
        visible = z < 0.0  # 在相机前方

        # 避免除零：对不可见点先占位
        inv_mz = np.zeros_like(z, dtype=np.float32)
        inv_mz[visible] = 1.0 / (-z[visible])

        u = np.full_like(z, np.nan, dtype=np.float32)
        v = np.full_like(z, np.nan, dtype=np.float32)
        u[visible] = fx * (x[visible] * inv_mz[visible]) + cx
        v[visible] = -fy * (y[visible] * inv_mz[visible]) + cy

        if image_size is not None and clip_outside:
            W, H = int(image_size[0]), int(image_size[1])
            in_img = (u >= 0.0) & (u < W) & (v >= 0.0) & (v < H)
            visible = visible & in_img

        out = {
            "u": u,
            "v": v,
            "visible": visible.astype(bool),
        }
        if return_depth:
            z_depth = np.full_like(z, np.nan, dtype=np.float32)
            z_depth[visible] = -z[visible]
            ray_depth = np.full_like(z, np.nan, dtype=np.float32)
            # 欧氏距离
            ray_depth[visible] = np.sqrt(
                x[visible] ** 2 + y[visible] ** 2 + z[visible] ** 2
            )
            out.update({"z_depth": z_depth, "ray_depth": ray_depth})
        return out

    # =========================
    # 静态：像素 -> 射线/三维点（针孔）
    # =========================

    @staticmethod
    def pixel_to_cam_ray_from_K(u: float, v: float, K: np.ndarray) -> np.ndarray:
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x = (u - cx) / fx
        y = -(v - cy) / fy      # 像素向下正 -> 相机 +Y 向上
        z = -1.0                # forward = -Z
        d = np.array([x, y, z], dtype=np.float32)
        return d / (np.linalg.norm(d) + 1e-12)

    @staticmethod
    def cam_ray_to_world(ray_cam: np.ndarray, T_world_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = T_world_cam[:3, :3]
        t = T_world_cam[:3, 3]
        d_world = R @ ray_cam
        d_world /= (np.linalg.norm(d_world) + 1e-12)
        return t.copy(), d_world

    @staticmethod
    def pixel_to_world_ray_static(u: float, v: float, K: np.ndarray, T_world_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d_cam = SensorGeometry.pixel_to_cam_ray_from_K(u, v, K)
        return SensorGeometry.cam_ray_to_world(d_cam, T_world_cam)

    @staticmethod
    def pixel_depth_to_world_static(
        u: float, v: float, depth: float, K: np.ndarray, T_world_cam: np.ndarray, convention: str = "ray"
    ) -> np.ndarray:
        if convention not in ("ray", "z"):
            raise ValueError("convention 仅支持 'ray' 或 'z'")
        R = T_world_cam[:3, :3]
        t = T_world_cam[:3, 3]
        if convention == "ray":
            o, d = SensorGeometry.pixel_to_world_ray_static(
                u, v, K, T_world_cam)
            return o + d * float(depth)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        z_cam = -float(depth)
        x_cam = -(u - cx) / fx * z_cam
        y_cam = (v - cy) / fy * z_cam
        p_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float32)
        return R @ p_cam + t

    @staticmethod
    def pixels_depths_to_world_batch_static(
        uv: np.ndarray, depths: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, convention: str = "ray"
    ) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float32)
        d = np.asarray(depths, dtype=np.float32).reshape(-1)
        assert uv.shape[0] == d.shape[0], "uv 与 depths 数量不一致"
        R = T_world_cam[:3, :3]
        t = T_world_cam[:3, 3]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if convention == "ray":
            x = (uv[:, 0] - cx) / fx
            y = -(uv[:, 1] - cy) / fy
            z = -np.ones_like(x)
            d_cam = np.stack([x, y, z], axis=1)
            d_cam /= (np.linalg.norm(d_cam, axis=1, keepdims=True) + 1e-12)
            d_world = (R @ d_cam.T).T
            d_world /= (np.linalg.norm(d_world, axis=1, keepdims=True) + 1e-12)
            pts = t[None, :] + d_world * d[:, None]
            return pts.astype(np.float32)
        elif convention == "z":
            z_cam = -d
            x_cam = (uv[:, 0] - cx) / fx * z_cam
            y_cam = -(uv[:, 1] - cy) / fy * z_cam
            p_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
            pts = (R @ p_cam.T).T + t[None, :]
            return pts.astype(np.float32)
        else:
            raise ValueError("convention 仅支持 'ray' 或 'z'")

    # =========================
    # 动态接口（从 env 取值后转调静态）
    # =========================
    def __init__(self, env, agent_name: str = "main_agent"):
        if hasattr(env, "env"):
            self._env = env.env
            self._ENV = env
        else:
            self._env = env
            self._ENV = None

        self._sim = self._env.sim
        self._cfg = getattr(env, "cfg", None)
        self._agent_name = agent_name
        if self._cfg is None:
            raise RuntimeError("需要可访问的 Habitat 配置 (env.cfg)")

    def list_sensors(self) -> List[str]:
        return list(self._sim.get_agent_state().sensor_states.keys())

    def get_T_world_agent(self) -> np.ndarray:
        return SensorGeometry.get_T_world_agent_static(self._sim.get_agent_state())

    def get_T_world_sensor(self, sensor_name: Optional[str] = None) -> np.ndarray:
        # 动态解析可用名，再调静态
        names = self.list_sensors()
        name = self._static_resolve_sensor_name(names, sensor_name)
        return SensorGeometry.get_T_world_sensor_static(self._sim.get_agent_state(), name)

    def get_K_from_cfg(self, sensor_name: Optional[str] = None) -> Dict[str, Any]:
        return SensorGeometry.get_K_from_cfg_static(self._cfg, sensor_name, self._agent_name)

    def _get_depth_at(self, u: int, v: int, depth_name: Optional[str] = "depth") -> float:
        """
        从最近一次观测中读取像素 (v,u) 的深度值（原始值，不做单位转换）。
        支持 obs['depth'] 或 obs[depth_name]；支持 HxW 或 HxWx1。
        """
        if self._ENV is None or self._ENV.observations is None:
            raise RuntimeError(
                "没有可用的最近一次观测：请先调用 env.reset()/env.step() 后再取深度。")

        obs_key_candidates = []
        if depth_name:
            obs_key_candidates.append(depth_name)
        obs_key_candidates.append("depth")  # 常见默认键

        obs = self._ENV.observations
        depth_img = None
        for k in obs_key_candidates:
            if k in obs:
                depth_img = obs[k]
                break
        if depth_img is None:
            raise KeyError(
                f"在最近一次观测中找不到深度键：候选 {obs_key_candidates}，可用键：{list(obs.keys())}")

        # HxW or HxWx1
        if depth_img.ndim == 3 and depth_img.shape[2] == 1:
            return float(depth_img[v, u, 0])
        return float(depth_img[v, u])

    def world_to_pixel(
        self,
        p_world: np.ndarray,
        sensor_name: Optional[str] = "depth",
        clip_outside: bool = True,
        return_depth: bool = True,
        return_concealed: bool = True,
        convention: str = "ray"
    ) -> Dict[str, Any]:
        """
        动态接口：从 env cfg/pose 读取 K 与 T_world_cam，投影世界点到像素。
        """
        # 取内参与外参
        Kinfo = self.get_K_from_cfg(sensor_name)
        T_ws = self.get_T_world_sensor(sensor_name)
        # 图像尺寸
        W, H = int(Kinfo["width"]), int(Kinfo["height"])
        result = SensorGeometry.world_to_pixel_static(
            p_world, Kinfo["K"], T_ws, image_size=(W, H),
            clip_outside=clip_outside, return_depth=return_depth or return_concealed
        )

        if not return_concealed:
            return result
        if result["visible"]:
            real_depth = self._get_depth_at(
                min(round(result["u"]), W - 1), min(round(result["v"]), H - 1), sensor_name)
            scfg = self._get_depth_cfg(
                self._cfg, sensor_name, self._agent_name)
            real_depth = self._depth_raw_to_meters(real_depth, scfg)
            if convention == "ray":
                result["concealed"] = result["ray_depth"] < real_depth + 0.05
            else:
                result["concealed"] = result["z_depth"] < real_depth + 0.05
        return result

    def world_points_to_pixels_batch(
        self,
        P_world: np.ndarray,
        sensor_name: Optional[str] = "depth",
        clip_outside: bool = True,
        return_depth: bool = True,
    ) -> Dict[str, Any]:
        """
        动态接口：批量世界点 -> 像素。
        """
        Kinfo = self.get_K_from_cfg(sensor_name)
        T_ws = self.get_T_world_sensor(sensor_name)
        W, H = int(Kinfo["width"]), int(Kinfo["height"])
        return SensorGeometry.world_points_to_pixels_batch_static(
            P_world, Kinfo["K"], T_ws, image_size=(W, H),
            clip_outside=clip_outside, return_depth=return_depth
        )

    # ---- 动态：像素 -> 射线/世界点 ----
    def pixel_to_world_ray(self, u: float, v: float, sensor_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        Kinfo = self.get_K_from_cfg(sensor_name)
        T_ws = self.get_T_world_sensor(sensor_name)
        return SensorGeometry.pixel_to_world_ray_static(u, v, Kinfo["K"], T_ws)

    def pixel_depth_to_world(
        self, u: float, v: float, depth: float, sensor_name: Optional[str] = None, convention: str = "ray"
    ) -> np.ndarray:
        Kinfo = self.get_K_from_cfg(sensor_name)
        T_ws = self.get_T_world_sensor(sensor_name)
        return self.pixel_depth_to_world_static(u, v, depth, Kinfo["K"], T_ws, convention)

    def pixels_depths_to_world_batch(
        self, uv: np.ndarray, depths: np.ndarray, sensor_name: Optional[str] = None, convention: str = "ray"
    ) -> np.ndarray:
        Kinfo = self.get_K_from_cfg(sensor_name)
        T_ws = self.get_T_world_sensor(sensor_name)
        return self.pixels_depths_to_world_batch_static(uv, depths, Kinfo["K"], T_ws, convention)

    def pixel_to_world(self, u: int, v: int, sensor_name: Optional[str] = "depth", convention: str = "ray") -> np.ndarray:
        """
        从当前帧的深度图中取 (u,v) 像素，自动根据 cfg 将深度值转为米，再反投影到世界坐标。
        convention:
        - "ray": 深度为沿射线的欧氏距离（Habitat 默认）
        - "z"  : 深度为相机坐标系 Z 分量（若你自己这样定义的话）
        """

        # 1) 取原始深度（可能是 0~1 的归一化，也可能已是米）
        d_raw = self._get_depth_at(u, v, sensor_name)

        # 2) 从 cfg 读取比例尺，转米
        scfg = self._get_depth_cfg(self._cfg, sensor_name, self._agent_name)
        d_m = self._depth_raw_to_meters(d_raw, scfg)

        # 3) 取内参与外参
        Kinfo = self.get_K_from_cfg(sensor_name)
        T_ws = self.get_T_world_sensor(sensor_name)

        # 4) 反投影到世界坐标
        return self.pixel_depth_to_world_static(u, v, d_m, Kinfo["K"], T_ws, convention)
