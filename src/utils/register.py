from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from dataclasses import dataclass, field
import importlib
import re
from threading import RLock


@dataclass(frozen=True)
class RegItem:
    name: str
    target: Any                                
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    meta: Dict[str, Any] = field(default_factory=dict)

    def all_names(self) -> Tuple[str, ...]:
        return (self.name, *self.aliases)


class Registry:
    """
    通用注册器：
      - register(name|None, aliases=(), override=False, **meta)
      - get(name) -> target
      - create(name, **kwargs) -> 实例（若 target 可调用）
      - list(pattern=None, meta_filter=None) -> [(name, RegItem)]
      - load_str("pkg.mod:attr") 延迟加载
    """
    def __init__(self, category: str):
        self._category = category
        self._lock = RLock()
        self._name2item: Dict[str, RegItem] = {}

                                
    def _check_name_free(self, name: str, override: bool):
        if not override and name in self._name2item:
            raise KeyError(f"[{self._category}] 名称已存在: {name}. 如需覆盖请传 override=True")

    def _index_item(self, item: RegItem, override: bool):
        for n in item.all_names():
            self._check_name_free(n, override)
        for n in item.all_names():
            self._name2item[n] = item

    def register(
        self,
        name: Optional[str] = None,
        *,
        aliases: Iterable[str] = (),
        override: bool = False,
        **meta: Any,
    ):
        """
        用法1（装饰器）:
            @registry.register()               # 使用被装饰对象.__name__ 作为 name
            @registry.register("my_name", aliases=("n1","n2"), task="cls")

        用法2（函数）:
            registry.register("name")(MyClass)  # 或 registry.add("name", MyClass)
        """
        def deco(target: Any):
            true_name = name or getattr(target, "__name__", None)
            if not true_name:
                raise ValueError("无法自动推断名称，请显式传入 name")
            item = RegItem(true_name, target, tuple(aliases), dict(meta))
            with self._lock:
                self._index_item(item, override)
            return target
        return deco

    def add(
        self,
        name: str,
        target: Any,
        *,
        aliases: Iterable[str] = (),
        override: bool = False,
        **meta: Any,
    ):
        """非装饰器方式注册"""
        return self.register(name, aliases=aliases, override=override, **meta)(target)

                                 
    def has(self, name: str) -> bool:
        return name in self._name2item

    def get_item(self, name: str) -> RegItem:
        try:
            return self._name2item[name]
        except KeyError:
            raise KeyError(f"[{self._category}] 未找到注册项: {name}")

    def get(self, name: str) -> Any:
        """返回 target（类/函数/可调用/对象）"""
        return self.get_item(name).target

    def meta(self, name: str) -> Dict[str, Any]:
        return dict(self.get_item(name).meta)

    def list(
        self,
        pattern: Optional[str] = None,
        meta_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, RegItem]:
        """
        pattern: 正则过滤名称
        meta_filter: {key: value} 完全匹配过滤（AND 关系）
        """
        with self._lock:
            items = {}
            for n, it in self._name2item.items():
                if n != it.name:            
                    continue
                if pattern and not re.search(pattern, n):
                    continue
                if meta_filter and not all(it.meta.get(k) == v for k, v in meta_filter.items()):
                    continue
                items[n] = it
            return items

                                    
    def create(self, name: str, /, *args, **kwargs) -> Any:
        """
        若 target 为类/函数/可调用：返回 target(*args, **kwargs)
        若 target 为字符串 "pkg.mod:attr"：动态 import 后再调用/返回
        若 target 为非可调用对象：直接返回
        """
        target = self.get(name)
        if isinstance(target, str):
            target = self.load_str(target)
        if callable(target):
            return target(*args, **kwargs)
        return target

    @staticmethod
    def load_str(spec: str) -> Any:
        """
        支持 "package.module:attr" 或 "package.module" 形式
        """
        if ":" in spec:
            mod, attr = spec.split(":", 1)
            m = importlib.import_module(mod)
            return getattr(m, attr)
        return importlib.import_module(spec)

model_registry   = Registry("model")
dataset_registry = Registry("dataset")
prompt_registry  = Registry("prompt")

env_registry = Registry("env")
agent_registry = Registry("agent")