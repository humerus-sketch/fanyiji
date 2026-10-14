# 第三方组件与模型

本程序独立实现，未复制 Sokuji 源码。构建包含以下第三方组件；发行目录 `licenses` 保存安装包提供的许可证文件，模型许可证保存在 `models` 对应子目录。安装包缺失的许可文本另保存在源码 `third_party_licenses`，构建复制到 `licenses/supplemental`；其 `MANIFEST.json` 记录固定官方来源、对应版本与 SHA-256，不包含本机路径或用户资料。

| 组件 | 官方来源 |
| --- | --- |
| PySide6 / Qt / shiboken6 | [Qt for Python](https://www.qt.io/qt-for-python)；本构建动态使用发行包中的 Qt DLL |
| NumPy | [NumPy](https://github.com/numpy/numpy) |
| sherpa-onnx / ONNX Runtime | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)、[ONNX Runtime](https://github.com/microsoft/onnxruntime) |
| CTranslate2 | [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) |
| SentencePiece | [google/sentencepiece](https://github.com/google/sentencepiece) |
| PyAudioWPatch / PortAudio | [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch)、[PortAudio](https://github.com/PortAudio/portaudio) |
| soxr / libsoxr | [python-soxr](https://github.com/dofuuz/python-soxr) |
| websocket-client 1.9.2 | [官方源码](https://github.com/websocket-client/websocket-client/)、[固定版本 PyPI](https://pypi.org/project/websocket-client/1.9.2/)，Apache-2.0 |
| PyInstaller 引导程序 | [PyInstaller](https://github.com/pyinstaller/pyinstaller) |

### 程序运行库补充许可

- **PySide6-Essentials / shiboken6 / Qt 6.12.0**：安装包声明 `LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only`。从同版本官方源码补齐 LGPL、GPL 和 Qt exception 文本，动态运行库保持独立文件。对应源代码和官方授权说明见补充目录的 `Qt/NOTICE.txt`。商业许可引用文件本身不授予商业许可。
- **CTranslate2 4.8.2 / SentencePiece 0.2.2**：分别保留固定版本官方 MIT / Apache-2.0 文本；SentencePiece 内置第三方许可一并保留。
- **websocket-client 1.9.2**：安装包 metadata 和 [官方固定版本说明](https://pypi.org/project/websocket-client/1.9.2/) 均声明 Apache-2.0；安装分发的 `websocket_client-1.9.2.dist-info/licenses/LICENSE` 含 Apache License 2.0 原文，构建按运行库许可收集流程保留。用于程序自有 Chrome 的本机 CDP WebSocket 通道，不以其许可证替代浏览器或平台的使用条款。
- **ONNX Runtime 1.28.2**：版本来自所分发 DLL 的版本资源；保留同版本官方 MIT 及 `ThirdPartyNotices.txt`。
- **PortAudio**：运行库报告 `V19.7.0-devel, revision unknown`；保留本程序使用的 PyAudioWPatch `v0.2.12.9` 源码中 `portaudio_v19/LICENSE.txt`，不把未知 revision 写成已确认的 PortAudio tag。
- **proxy-tools 0.1.0**：安装 metadata 和官方 `setup.py` 声明 MIT，但官方固定 commit 的 `LICENSE.txt` 实际含 BSD 条文；原文完整保留，未擅自将其改写成 MIT。该来源差异记录在补充目录的 manifest 中。
- **Intel OpenMP / NVIDIA cuDNN**：CTranslate2 还分发独立原生 DLL。Intel OpenMP build `5.0.20250910` 与官方 `intel-openmp 2025.3.0` wheel 内 DLL 的 SHA-256 相同，保留该 wheel 的 Intel Developer Tools EULA 原文及 UTF-8 阅读副本；cuDNN `9.10.2.21` 保留 NVIDIA 固定 `v9.10.2` 文档中的 EULA、cuDNN Supplement 与第三方 acknowledgements 的 HTML 原文和阅读文本。它们适用各自条款，不能以 CTranslate2 MIT 或 LLVM OpenMP 许可代替。

这些文件用于保留已识别运行库的上游授权与归属信息，不构成法律合规保证；模型权重许可与程序运行库许可分别适用。

## 模型

- **SenseVoiceSmall ONNX INT8**：使用 [sherpa-onnx 转换资产](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17)，固定 revision 为 `2365baeacb507f821a0c8120fcee3d484dba7a07`。原始 `LICENSE` 指向 [FunASR 模型授权条款](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)，条款副本另存 `models/asr/MODEL_LICENSE`；模型授权与推理框架授权分别保留。
- **M2M100 418M**：来自 [Meta 模型页](https://huggingface.co/facebook/m2m100_418M)，固定 revision 为 `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636`。模型页标注 MIT；转换为 CPU INT8，保留原词表、模型卡和 `models/translation/LICENSE`。
- **Silero VAD**：模型来自 [sherpa-onnx 官方资产](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)，[Silero MIT LICENSE](https://github.com/snakers4/silero-vad/blob/master/LICENSE) 保存在 `models/vad/LICENSE`。资产以 SHA256 固定于 manifest。
- **讲话筛选 Zipformer Small INT8**：来自 [k2-fsa 转换模型](https://huggingface.co/k2-fsa/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15)，固定 revision `c24f9d0fdf7d0b8c0c4f9733aa0cac51fda10c95`；固定模型卡声明 Apache-2.0。模型与 AudioSet 类别 CSV 的大小和 SHA-256 固定于 `models_download.py`，模型卡、SOURCE 和 Apache-2.0 通用许可文本保留。分类不能当作歌声分离或测得准确率。CED 仅在开发目录评测。
- **自测音频**：两份 WAV 来自同一 sherpa-onnx ASR 资产中的 `test_wavs/zh.wav`、`test_wavs/ja.wav`，仅用于开发和本地自测。

构建的权重、资产及许可文件哈希见 `models/manifest.json`。以上来源说明用于追踪本次本地构建，不改变任何上游授权条款。

## 外置翻译模型与原生推理

新版程序包不包含模型权重，模型由用户通过管理窗口下载，或在其他电脑下载后复制；官方许可与权重许可各自适用。已下载翻译模型保留；生产识别仅使用 SenseVoice。

- **GTCRN 语音增强**：作者 [GTCRN 仓库](https://github.com/Xiaobin-Rong/gtcrn) 的 [LICENSE](https://github.com/Xiaobin-Rong/gtcrn/blob/main/LICENSE) 声明 MIT。使用 [sherpa-onnx 语音增强发行资产](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models) `gtcrn_simple.onnx`；大小 535,638 bytes，SHA256 `e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534`。该哈希由官方发行下载计算，非发布者签名。权重外置，不随程序包分发。
- **可选 UVR MDXNET 9482 伴奏分离**：使用 [sherpa-onnx 官方模型文档](https://k2-fsa.github.io/sherpa/onnx/source-separation/models.html) 与 [发行资产](https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models) 的 `UVR_MDXNET_9482.onnx`；大小 29,704,738 bytes，SHA256 `9d78f8566fa8198065214ab628be1de966a500c57786695aa4b13e2b27a7727d`，与发行 digest 一致。上游权重源 [TRvlvr/model_repo](https://github.com/TRvlvr/model_repo) 的顶层未提供 LICENSE；具体权重使用条款仍需向发布者核实，不能用推理框架许可代替模型许可。权重仅供用户另行获取，不随程序包分发。

| 项目 | 官方来源与固定版本 |
| --- | --- |
| llama.cpp | [官方 b11429](https://github.com/ggml-org/llama.cpp/tree/b11429)，MIT；CPU/CUDA 原生运行库随程序提供 |
| LLVM OpenMP | [LLVM 许可](https://github.com/llvm/llvm-project/blob/main/openmp/LICENSE.TXT)，发行 runtime 内保留许可 |
| NVIDIA CUDA runtime / cuBLAS | [CUDA Toolkit 许可](https://docs.nvidia.com/cuda/eula/index.html)，发行 runtime 内保留来源与许可；不含 NVIDIA 显卡驱动 |
| Hy-MT2 1.8B / 7B / 30B-A3B GGUF | [腾讯官方模型](https://huggingface.co/tencent)，各候选固定 revision 和 SHA 位于 `models_download.py`；官方仓库 Apache-2.0 |
| Qwen3 1.7B GGUF | [Qwen 官方模型](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF)，固定 revision `90862c4b9d2787eaed51d12237eafdfe7c5f6077`；Apache-2.0 |

程序包 `licenses/models` 可保留模型许可文本，许可文本不是模型权重；本地下载资产与对应来源、SHA、LICENSE 另存外置模型目录。上述说明不改变上游授权条款。

历史识别评测保留 [OpenAI Whisper](https://github.com/openai/whisper) 与 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的报告、来源和适配工具，分别适用 MIT 与 Apache-2.0。它们的本地识别权重已按本次清理要求删除，不能在生产程序中下载、导入或切换。历史评测工具的固定版本和 SHA 仅用于追踪旧实验。
# 内置浏览器依赖

- **pywebview 6.2.1**：[官方源码](https://github.com/r0x0r/pywebview/tree/6.2.1)，BSD-3-Clause；Windows 使用 WinForms + WebView2 后端，保留发行包内 DLL、JS 及相应许可。
- **pythonnet / clr-loader / cffi / proxy-tools / bottle / typing-extensions / pycparser**：Python 与 .NET 互操作和 pywebview 依赖，发行包 `licenses` 保留各安装分发的许可文本。
- **Microsoft WebView2 SDK / Evergreen Runtime**：[官方分发说明](https://learn.microsoft.com/microsoft-edge/webview2/concepts/distribution)。SDK loader 与 .NET assemblies 随 pywebview 提供；Evergreen 浏览器 Runtime 外部安装，使用 Microsoft 许可和更新机制，不把浏览器账号资料或 Runtime 安装目录复制到程序包。

## Chrome 原生字幕模式

Google Chrome 使用用户自行安装的官方浏览器，不随翻译姬分发。专用模式使用独立的外置资料目录；可选的现有 Chrome 接入仅做只读字幕采集，不复制个人登录资料。原生字幕与实时翻译由 Chrome 提供，字幕气泡的可访问结构不属于稳定对外 API；透明隐藏、恢复和持续读取的兼容性仍需实机验证。上述依赖许可不授予平台发送权限，也不表示已经验证真实弹幕投递或字幕准确率。
