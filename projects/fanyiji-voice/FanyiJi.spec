"""Portable Windows application; model weights are downloaded separately."""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

root = Path(SPECPATH)
native_packages = ("sherpa_onnx", "ctranslate2", "sentencepiece", "pyaudiowpatch", "soxr")
binaries = []
datas = []
datas += collect_data_files("webview", include_py_files=False)
for package in native_packages:
    binaries += collect_dynamic_libs(package)
    datas += collect_data_files(package, include_py_files=False)

a = Analysis(
    [str(root / "run.py")],
    pathex=[str(root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "numpy", "sherpa_onnx", "sherpa_onnx.lib._sherpa_onnx",
        "ctranslate2", "ctranslate2._ext", "sentencepiece", "sentencepiece._sentencepiece",
        "pyaudiowpatch", "_portaudiowpatch", "soxr", "soxr.soxr_ext",
        "webview.platforms.winforms", "webview.platforms.edgechromium", "clr", "clr_loader",
    ],
    excludes=["torch", "transformers", "huggingface_hub", "pytest", "tkinter", "matplotlib",
              "webview.platforms.qt", "webview.platforms.cef", "webview.platforms.gtk", "webview.platforms.cocoa",
              "ctranslate2.converters", "ctranslate2.specs", "fanyiji.chrome_caption_worker", "fanyiji.chrome_browser_service"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="翻译姬·语音版",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
    version=str(root / "version_info.txt"),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="翻译姬·语音版")
