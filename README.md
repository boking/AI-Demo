## AI 漫剧 Demo

一个使用 Python + FastAPI 搭建的 AI 漫剧小产品 Demo（骨架版），后续可以接入通用 / 多模态大模型与 TTS。

### 当前目标

- 提供默认4个场景，生成一集「漫剧分镜」，自动播放
- 提供一个简单前端页面：调用该接口，并以卡片形式展示分镜（图占位 + 文本字幕）。
- 用户可以随时打断，可以选择不同的分支，故事走向不同的结局

### 运行方式

1. 创建虚拟环境并安装依赖：

```bash
cd /Users/bcui/Downloads/vscode
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
pip install -r requirements.txt
```

2. 启动后端：

```bash
uvicorn app.main:app --reload
点击frontend.html,进入到web页面
```

3. 访问前端，体验简单的 AI 漫剧分镜生成。

> 说明：目前代码中的“大模型调用”会以占位实现（假数据），方便先把产品逻辑和数据结构跑通，再接真实模型。

