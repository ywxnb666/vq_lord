from pathlib import Path
import importlib.machinery
import importlib.util
import sys


def load_compiled_module(module_name: str):
    pycache_dir = Path(__file__).resolve().parent.parent / '__pycache__'
    version_tag = f'cpython-{sys.version_info.major}{sys.version_info.minor}'
    candidates = [
        pycache_dir / f'{module_name}.{version_tag}.pyc',
        pycache_dir / f'{module_name}.cpython-310.pyc',
        pycache_dir / f'{module_name}.cpython-38.pyc',
    ]
    for candidate in candidates:
        if candidate.exists():
            loader = importlib.machinery.SourcelessFileLoader(module_name, str(candidate))
            spec = importlib.util.spec_from_loader(module_name, loader)
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            return module
    raise ModuleNotFoundError(f'No compiled module available for {module_name!r} in {pycache_dir}')
