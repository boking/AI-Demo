from typing import List, Optional
import json
import logging
import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests


logger = logging.getLogger("ai-manga")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI 漫剧 Demo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EpisodeRequest(BaseModel):
    """用户发起一集漫剧生成时的请求体"""

    prompt: str = Field(..., description="一句话设定，例如：一个倒霉的实习生在第一天上班就撞见鬼")
    world_view: Optional[str] = Field(
        default=None, description="世界观 / 风格，例如：校园、赛博、武侠"
    )
    tone: Optional[str] = Field(
        default=None, description="基调，例如：搞笑、治愈、悬疑、热血"
    )


class BranchRequest(BaseModel):
    """用户在分支点做出选择后，请求生成后续分镜"""

    prompt: str = Field(..., description="与初始一集相同的一句话设定")
    world_view: Optional[str] = Field(default=None, description="世界观 / 风格")
    tone: Optional[str] = Field(default=None, description="基调")
    choice: str = Field(..., description="用户在结局分支时做出的选择内容")
    story_outline: Optional[str] = Field(
        default=None,
        description="当前已播故事的简要梗概（由前端拼接），有助于模型延续剧情",
    )
    choice_history: Optional[List[str]] = Field(
        default=None,
        description="用户之前做过的所有选择历史，用于生成连贯的多分支故事",
    )
    branch_depth: Optional[int] = Field(
        default=1,
        description="当前分支深度（第几次选择），用于判断是否接近结局",
    )


class MultiBranchRequest(BaseModel):
    """请求生成多个分支路径和不同结局"""

    prompt: str = Field(..., description="初始设定")
    world_view: Optional[str] = Field(default=None)
    tone: Optional[str] = Field(default=None)
    current_path: List[str] = Field(
        default_factory=list,
        description="当前选择路径（用户做过的所有选择）",
    )
    target_outcome: Optional[str] = Field(
        default=None,
        description="目标结局类型，例如：'恶人有恶报'、'恶人活到最后'、'主角牺牲'等",
    )


class VoiceRoundRequest(BaseModel):
    """对话轮次请求：用户说一句话，AI 角色给出语音回复脚本"""

    utterance: str = Field(..., description="用户本轮说的话（文本形式）")
    world_view: Optional[str] = Field(default=None, description="世界观 / 风格")
    tone: Optional[str] = Field(default=None, description="基调")
    history: Optional[List[dict]] = Field(
        default=None,
        description="之前的对话历史，用于让角色“记得你”，格式为 [{role: 'user'|'assistant', content: '...'}]",
    )


class VoiceRoundResponse(BaseModel):
    """对话轮次返回：一段角色回复文本，可用于 TTS 播报"""

    reply: str = Field(..., description="AI 角色本轮的回复台词")
    speaker: str = Field(
        default="女主",
        description="当前说话角色名称（默认是“女主”，可根据设定扩展）",
    )
    mood: Optional[str] = Field(
        default=None, description="情绪标签，例如：冷静、愤怒、玩笑"
    )


class Frame(BaseModel):
    """单格分镜"""

    index: int = Field(..., description="第几格，从 1 开始")
    image_prompt: str = Field(
        ..., description="用于生成该格图片的提示词（后续可直接喂给多模态模型）"
    )
    dialogue: str = Field(..., description="该格中角色说的话或主要台词")
    speaker: str = Field(..., description="说话的角色名称（或“旁白”）")
    mood: Optional[str] = Field(
        default=None, description="情绪标签，例如：紧张、搞笑、温馨"
    )
    narration: Optional[str] = Field(
        default=None, description="补充说明/旁白，用于字幕或 TTS 旁白"
    )


class Episode(BaseModel):
    """一集漫剧，由多格分镜组成"""

    episode_id: str
    title: str
    description: str
    frames: List[Frame]
    choices: Optional[List[str]] = Field(
        default=None,
        description="在当前集末尾提供给用户的结局分支选项文案（例如两到三个选项）",
    )


class BranchEpisode(BaseModel):
    """分支选择后的后续分镜（包含追加的分镜和新的选择点）"""

    episode_id: str
    choice: str
    frames: List[Frame]
    choices: Optional[List[str]] = Field(
        default=None,
        description="新的选择点（如果这是最后一次选择，则为空）",
    )


def _build_base_title(prompt: str) -> str:
    return "AI 漫剧 · " + (prompt[:20] if prompt else "神秘故事")


def _build_description(payload: EpisodeRequest) -> str:
    return (
        f"根据你的设定生成的一集短漫。世界观：{payload.world_view or '默认都市'}，"
        f"基调：{payload.tone or '轻悬疑'}。"
    )


def _generate_mock_frames() -> List[Frame]:
    """后备方案：如果大模型不可用，使用固定的示例分镜。"""
    return [
        Frame(
            index=1,
            image_prompt="办公室早晨，主角打着哈欠走进开放工位，屏幕发出诡异蓝光",
            speaker="旁白",
            mood="日常",
            dialogue="第一天上班，他完全没想到今天会遇到一件‘超自然事故’。",
            narration=None,
        ),
        Frame(
            index=2,
            image_prompt="主角坐到位置上，电脑自动开机，聊天窗口自己弹出",
            speaker="电脑里的陌生人",
            mood="诡异",
            dialogue="“嗨，新人，你能听到我说话吗？”",
            narration="聊天窗口没有任何头像，只有一行行自己跳出来的文字。",
        ),
        Frame(
            index=3,
            image_prompt="主角瞪大眼睛，身后同事一脸疲惫，对这一切浑然不觉",
            speaker="主角",
            mood="慌张",
            dialogue="“谁在整我？这是什么恶作剧？”",
            narration=None,
        ),
        Frame(
            index=4,
            image_prompt="屏幕上浮现出旧公司系统的界面，充满红色警告与闪烁的错误提示",
            speaker="电脑里的陌生人",
            mood="紧张",
            dialogue="“别大声，我只剩今天可以和你说话。服务器里…有东西被关错了。”",
            narration="屏幕上闪过的，是他从未见过的内网系统。",
        ),
        Frame(
            index=5,
            image_prompt="会议室门被推开，领导招手示意主角进去",
            speaker="旁白",
            mood="压迫",
            dialogue="这时，直属领导突然叫他进会议室，神情严肃。",
            narration="电脑屏幕却还在疯狂闪烁提示：『不要进去。』",
        ),
        Frame(
            index=6,
            image_prompt="特写主角的脸，一边是会议室门，一边是闪烁的电脑屏幕",
            speaker="旁白",
            mood="抉择",
            dialogue="第一天上班，他要在‘正常人生’和‘诡异求救’之间做出第一个选择。",
            narration="无论他怎么选，这一集的故事，才刚刚开始……",
        ),
    ]


def _generate_frames_with_llm(payload: EpisodeRequest) -> dict:
    """
    使用通用大模型生成 6–8 格分镜。
    这里通过 HTTP 调用兼容 OpenAI 协议的 chat/completions 接口（支持 OpenRouter 等网关），避免 SDK 兼容性问题。
    """
    # 你可以使用 OpenRouter 的 Key（推荐设置为 OPENROUTER_API_KEY），
    # 也可以继续使用 OPENAI_API_KEY；优先读取前者。
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.info("OPENROUTER_API_KEY / OPENAI_API_KEY 未配置，回退到 mock 分镜。")
        return {"frames": _generate_mock_frames(), "choices": []}

    system_prompt = (
        "你是一个擅长多分支互动故事的漫画编剧助手。\n"
        "根据用户提供的一句话设定、世界观和基调，生成一个完整的故事开头段落。\n"
        "\n重要要求：\n"
        "- 不要拘泥于固定的分镜数量，根据故事需要自然决定长度（建议 8-15 格）。\n"
        "- 这是一个完整的故事开头，要有足够的篇幅来：\n"
        "  1. 建立世界观和氛围\n"
        "  2. 介绍主要角色和他们的关系\n"
        "  3. 展现核心冲突或悬念\n"
        "  4. 在结尾设置第一个选择点，让用户决定故事走向\n"
        "- 故事要有深度和戏剧张力，不能只是简单的场景描述。\n"
        "- 在结尾给出 2~3 个可供用户选择的剧情分支选项。\n"
        "\n输出一个 JSON，形如：\n"
        "{\n"
        "  \"frames\": [\n"
        "    {\"index\":1, \"image_prompt\":\"...\", \"speaker\":\"...\", \"mood\":\"...\", \"dialogue\":\"...\", \"narration\":\"...\"},\n"
        "    ...\n"
        "  ],\n"
        "  \"choices\": [\"走进会议室\", \"留在工位继续调查\"]\n"
        "}\n"
        "\n注意：\n"
        "- index 从 1 开始递增。\n"
        "- image_prompt 要描述画面细节，便于之后用作图片生成提示词。\n"
        "- 对话尽量简洁有画面感，符合给定的世界观和基调。\n"
        "- choices 要是适合当前故事结尾的简短选项文案，这些选择将决定故事的后续发展。\n"
        "- 确保生成足够的分镜数量，让故事开头完整、有吸引力。\n"
    )

    user_prompt = {
        "prompt": payload.prompt,
        "world_view": payload.world_view or "默认都市",
        "tone": payload.tone or "轻悬疑",
    }

    try:
        # 你也可以把模型名放到环境变量里，例如 LLM_MODEL_NAME
        # 使用 OpenRouter 时，模型名类似于：openai/gpt-4o-mini、google/gemini-1.5-pro 等
        model_name = os.getenv("LLM_MODEL_NAME", "openai/gpt-4o-mini")

        # 默认使用 OpenRouter 的网关地址；如果你想直接连 OpenAI，可以把
        # OPENAI_API_BASE 设置为 https://api.openai.com/v1/chat/completions
        base_url = os.getenv(
            "OPENAI_API_BASE", "https://openrouter.ai/api/v1/chat/completions"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter 建议额外加的头，便于统计来源（可选）
        referer = os.getenv("APP_PUBLIC_URL", "http://localhost:8000")
        headers["HTTP-Referer"] = referer
        app_title = os.getenv("APP_NAME", "AI Manga Demo")
        headers["X-Title"] = app_title

        resp = requests.post(
            base_url,
            headers=headers,
            json={
                "model": model_name,
                "temperature": 0.9,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "请基于以下设定生成分镜 JSON（只输出 JSON）：\n"
                        + json.dumps(user_prompt, ensure_ascii=False),
                    },
                ],
            },
            timeout=30,
        )

        resp.raise_for_status()
        data_raw = resp.json()
        content = (
            data_raw.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        # 兼容模型可能包裹 ```json ... ```
        content_stripped = content.strip()
        if content_stripped.startswith("```"):
            content_stripped = content_stripped.strip("`")
            # 去掉可能的语言标记
            if content_stripped.startswith("json"):
                content_stripped = content_stripped[4:]

        data_json = json.loads(content_stripped)
        frames_data = data_json.get("frames", [])
        choices_data = data_json.get("choices") or []

        frames: List[Frame] = []
        for i, f in enumerate(frames_data, start=1):
            try:
                frames.append(
                    Frame(
                        index=f.get("index", i),
                        image_prompt=f.get("image_prompt", ""),
                        speaker=f.get("speaker", "旁白"),
                        mood=f.get("mood"),
                        dialogue=f.get("dialogue", ""),
                        narration=f.get("narration"),
                    )
                )
            except Exception as e:  # 单格解析失败不中断整个流程
                logger.warning("解析某一格分镜失败: %s, data=%s", e, f)

        # 安全兜底：如果解析后没有有效分镜，回退到 mock
        if not frames:
            logger.warning("大模型返回的分镜为空，回退到 mock 分镜。")
            return {"frames": _generate_mock_frames(), "choices": []}

        # 不限制分镜数量，使用AI生成的所有分镜，确保故事完整
        logger.info("生成的分镜数量: %d", len(frames))

        # choices 如果不是字符串数组，做一层清洗
        cleaned_choices: List[str] = []
        if isinstance(choices_data, list):
            for c in choices_data:
                if isinstance(c, str):
                    cleaned_choices.append(c.strip())
                elif isinstance(c, dict) and "text" in c:
                    cleaned_choices.append(str(c["text"]).strip())
        # 至少保证有 2 个可选项；不够就补默认的
        if len(cleaned_choices) < 2:
            cleaned_choices = cleaned_choices + [
                "顺从表面的安排，继续按照‘正常’的人生轨迹走下去",
                "跟随直觉，追查那些诡异而危险的信号",
            ]
            # 去重并保留顺序
            dedup = []
            for c in cleaned_choices:
                if c and c not in dedup:
                    dedup.append(c)
            cleaned_choices = dedup[:3]

        logger.info("LLM 生成的分支选项: %s", cleaned_choices)

        return {"frames": frames, "choices": cleaned_choices}
    except requests.exceptions.HTTPError as e:
        # 处理 HTTP 错误（401, 403, 429 等）
        status_code = e.response.status_code if hasattr(e, 'response') else None
        logger.error("调用大模型生成分镜失败（HTTP %s），回退到 mock 分镜: %s", status_code, e)
        return {"frames": _generate_mock_frames(), "choices": []}
    except Exception as e:
        logger.error("调用大模型生成分镜失败，回退到 mock 分镜: %s", e, exc_info=True)
        return {"frames": _generate_mock_frames(), "choices": []}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/generate_episode", response_model=Episode)
async def generate_episode(payload: EpisodeRequest) -> Episode:
    """
    生成一集 6–8 格分镜的短漫。
    优先调用通用大模型生成，若不可用则回退到内置 mock 分镜。
    """
    base_title = _build_base_title(payload.prompt)

    llm_result = _generate_frames_with_llm(payload)
    frames: List[Frame] = llm_result.get("frames", [])  # type: ignore[assignment]
    choices: List[str] = llm_result.get("choices", [])  # type: ignore[assignment]

    return Episode(
        episode_id="demo-001",
        title=base_title,
        description=_build_description(payload),
        frames=frames,
        choices=choices or None,
    )


def _generate_branch_frames_with_llm(payload: BranchRequest) -> dict:
    """
    基于已有故事梗概和用户的分支选择，生成后续分镜。
    根据分支深度决定生成长度：每次选择后生成完整的故事段落，并在结尾设置新的选择点（最后一次除外）。
    返回 {"frames": [...], "choices": [...]}
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.info("OPENROUTER_API_KEY / OPENAI_API_KEY 未配置，分支回退到 mock。")
        # 简单返回一段固定分支，避免出错
        return [
            Frame(
                index=1,
                image_prompt="特写主角的背影，他迈出一步，踏入昏暗的走廊",
                speaker="旁白",
                mood="紧张",
                dialogue=f"最终，他选择了：{payload.choice}。",
                narration="走廊尽头的门缓缓关上，一切再也回不到从前。",
            )
        ]

    branch_depth = payload.branch_depth or 1
    
    # 构建选择历史描述
    choice_history_text = ""
    if payload.choice_history and len(payload.choice_history) > 0:
        choice_history_text = "\n用户之前做过的选择：\n" + "\n".join(
            f"- 第{i+1}次选择：{choice}" for i, choice in enumerate(payload.choice_history)
        )
        choice_history_text += f"\n当前是第{branch_depth}次选择。"
    
    # 根据分支深度决定故事段落类型
    if branch_depth == 1:
        story_segment_hint = (
            "这是第一次选择后的故事发展。请生成一个完整的故事段落（6-12格），"
            "让故事有足够的发展空间，展现选择带来的后果和新的冲突。"
            "这个段落应该是一个完整的故事单元，有起承转合，并在结尾设置新的选择点。"
        )
    elif branch_depth == 2:
        story_segment_hint = (
            "这是第二次选择后的故事发展。请生成一个完整的故事段落（6-12格），"
            "继续推进剧情，深化冲突，让故事走向高潮。"
            "这个段落应该展现故事的核心矛盾，并在结尾设置最后一次选择点。"
        )
    else:  # branch_depth >= 3
        story_segment_hint = (
            "这是最后一次选择，故事即将结束。请生成一个完整、有冲击力的结局（8-15格），"
            "根据用户的所有选择自然发展，形成一个令人印象深刻的结局。"
            "结局可以是任何走向：'恶人有恶报'、'恶人活到最后'、'主角牺牲'、'开放式结局'等，"
            "只要符合故事逻辑和用户的选择路径。结局要有足够的篇幅来展现故事的完整收尾。"
        )
    
    system_prompt = (
        "你是一个擅长多分支故事的漫画编剧助手。\n"
        "用户已经听完了一段短篇故事，现在在分支点做出了一个选择。\n"
        f"{story_segment_hint}\n"
        f"{choice_history_text}\n"
        "\n重要要求：\n"
        "- 不要拘泥于固定的分镜数量，根据故事需要自然决定长度。\n"
        "- 每次选择后要生成一个完整的故事段落，有起承转合，有足够的戏剧张力。\n"
        "- 故事应该根据用户的所有选择历史自然发展，不能突兀。\n"
        "- 确保故事连贯、完整，每次生成的内容都要是一个完整的故事单元。\n"
        "\n输出一个 JSON，对象形如：\n"
        "{\n"
        "  \"frames\": [\n"
        "    {\"index\":1, \"image_prompt\":\"...\", \"speaker\":\"...\", \"mood\":\"...\", \"dialogue\":\"...\", \"narration\":\"...\"},\n"
        "    ...\n"
        "  ],\n"
        "  \"choices\": [\"选项1\", \"选项2\"]\n"
        "}\n"
        "\n注意：\n"
        "- 如果这是第1次或第2次选择，必须在结尾设置新的选择点（choices字段，2-3个选项）。\n"
        "- 如果这是第3次或更多次选择（最后一次），不要返回choices字段（或返回空数组），因为故事已经结束。\n"
        "\n要求：\n"
        "- index 从 1 开始即可（前端会自行重新编号）。\n"
        "- 续写要与 story_outline 的剧情和人物保持连贯。\n"
        "- 对话与画面要契合 choice 所代表的方向。\n"
        "- 生成足够的分镜数量，确保故事完整、有深度。\n"
    )

    user_prompt = {
        "prompt": payload.prompt,
        "world_view": payload.world_view or "默认都市",
        "tone": payload.tone or "轻悬疑",
        "choice": payload.choice,
        "story_outline": payload.story_outline or "",
    }

    try:
        model_name = os.getenv("LLM_MODEL_NAME", "openai/gpt-4o-mini")
        base_url = os.getenv(
            "OPENAI_API_BASE", "https://openrouter.ai/api/v1/chat/completions"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("APP_PUBLIC_URL")
        if referer:
            headers["HTTP-Referer"] = referer
        app_title = os.getenv("APP_NAME", "AI Manga Demo")
        headers["X-Title"] = app_title

        resp = requests.post(
            base_url,
            headers=headers,
            json={
                "model": model_name,
                "temperature": 0.9,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "请基于以下设定生成分支后续分镜 JSON（只输出 JSON）：\n"
                        + json.dumps(user_prompt, ensure_ascii=False),
                    },
                ],
            },
            timeout=30,
        )

        resp.raise_for_status()
        data_raw = resp.json()
        content = (
            data_raw.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        content_stripped = content.strip()
        if content_stripped.startswith("```"):
            content_stripped = content_stripped.strip("`")
            if content_stripped.startswith("json"):
                content_stripped = content_stripped[4:]

        data_json = json.loads(content_stripped)
        frames_data = data_json.get("frames", [])
        choices_data = data_json.get("choices") or []

        frames: List[Frame] = []
        for i, f in enumerate(frames_data, start=1):
            try:
                frames.append(
                    Frame(
                        index=f.get("index", i),
                        image_prompt=f.get("image_prompt", ""),
                        speaker=f.get("speaker", "旁白"),
                        mood=f.get("mood"),
                        dialogue=f.get("dialogue", ""),
                        narration=f.get("narration"),
                    )
                )
            except Exception as e:
                logger.warning("解析某一格分支分镜失败: %s, data=%s", e, f)

        if not frames:
            logger.warning("大模型返回的分支分镜为空，使用简单固定分支。")
            return {
                "frames": [
                    Frame(
                        index=1,
                        image_prompt="主角站在灯光与阴影的交界处，露出复杂的表情",
                        speaker="旁白",
                        mood="复杂",
                        dialogue=f"无论选择如何，他都清楚，今天发生的一切已经改变了他的人生轨迹。",
                        narration=None,
                    )
                ],
                "choices": []
            }

        # 清洗choices
        cleaned_choices: List[str] = []
        if isinstance(choices_data, list):
            for c in choices_data:
                if isinstance(c, str) and c.strip():
                    cleaned_choices.append(c.strip())
        
        # 如果是最后一次选择（branch_depth >= 3），不应该有choices
        if payload.branch_depth and payload.branch_depth >= 3:
            cleaned_choices = []

        # 不限制分镜数量，使用AI生成的所有分镜，确保故事完整
        logger.info("生成的分支分镜数量: %d, 选择点数量: %d", len(frames), len(cleaned_choices))
        return {"frames": frames, "choices": cleaned_choices}
    except Exception as e:
        logger.error("调用大模型生成分支分镜失败，回退到简单固定分支: %s", e, exc_info=True)
        # 根据分支深度决定是否返回选择点
        mock_choices = []
        if payload.branch_depth and payload.branch_depth < 3:
            mock_choices = ["继续当前路径", "尝试其他方法"]
        
        return {
            "frames": [
                Frame(
                    index=1,
                    image_prompt="主角转身离开屏幕，身后微弱的蓝光逐渐熄灭",
                    speaker="旁白",
                    mood="余韵",
                    dialogue=f"他做出了{payload.choice}，而这个选择的后果，将在未来一一显现。",
                    narration=None,
                )
            ],
            "choices": mock_choices
        }


@app.post("/generate_branch_episode", response_model=BranchEpisode)
async def generate_branch_episode(payload: BranchRequest) -> BranchEpisode:
    """
    在用户选择某个结局分支后，生成后续分镜。
    根据分支深度决定生成长度：深度浅生成 2-3 格，深度深生成 4-6 格形成完整结局。
    返回的 frames 只包含新追加的部分，前端负责与原始 episode 拼接。
    """
    result = _generate_branch_frames_with_llm(payload)
    frames: List[Frame] = result.get("frames", [])  # type: ignore[assignment]
    choices: List[str] = result.get("choices", [])  # type: ignore[assignment]
    
    return BranchEpisode(
        episode_id="demo-001-branch",
        choice=payload.choice,
        frames=frames,
        choices=choices if choices else None,
    )


@app.post("/generate_alternative_ending")
async def generate_alternative_ending(payload: MultiBranchRequest):
    """
    根据不同的选择路径，生成完全不同的结局。
    例如：如果用户选择了"恶人有恶报"路径，可以生成"恶人活到最后"的相反结局。
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "error": "需要配置 API key 才能生成多结局",
            "endings": []
        }
    
    # 构建当前路径描述
    path_text = "用户的选择路径：\n" + "\n".join(
        f"- 第{i+1}次：{choice}" for i, choice in enumerate(payload.current_path)
    ) if payload.current_path else "这是故事的开始。"
    
    target_outcome = payload.target_outcome or "根据故事逻辑自然发展"
    
    system_prompt = (
        "你是一个擅长创作多分支、多结局故事的漫画编剧。\n"
        f"当前故事设定：{payload.prompt}\n"
        f"世界观：{payload.world_view or '默认'}\n"
        f"基调：{payload.tone or '默认'}\n"
        f"{path_text}\n"
        f"\n请生成一个与当前路径不同的结局（目标：{target_outcome}）。\n"
        "生成 4-6 格分镜，形成一个完整、有冲击力的结局。\n"
        "输出 JSON：\n"
        "{\n"
        '  "frames": [\n'
        '    {"index":1, "image_prompt":"...", "speaker":"...", "mood":"...", "dialogue":"...", "narration":"..."},\n'
        "    ...\n"
        "  ],\n"
        '  "ending_type": "结局类型，例如：恶人有恶报、恶人活到最后、主角牺牲、开放式等"\n'
        "}\n"
    )
    
    try:
        model_name = os.getenv("LLM_MODEL_NAME", "openai/gpt-4o-mini")
        base_url = os.getenv(
            "OPENAI_API_BASE", "https://openrouter.ai/api/v1/chat/completions"
        )
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("APP_PUBLIC_URL", "http://localhost:8000")
        headers["HTTP-Referer"] = referer
        app_title = os.getenv("APP_NAME", "AI Manga Demo")
        headers["X-Title"] = app_title
        
        resp = requests.post(
            base_url,
            headers=headers,
            json={
                "model": model_name,
                "temperature": 0.95,  # 更高的温度，让结局更发散
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "请生成一个不同的结局 JSON（只输出 JSON）："
                    },
                ],
            },
            timeout=30,
        )
        
        resp.raise_for_status()
        data_raw = resp.json()
        content = (
            data_raw.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        
        content_stripped = content.strip()
        if content_stripped.startswith("```"):
            content_stripped = content_stripped.strip("`")
            if content_stripped.startswith("json"):
                content_stripped = content_stripped[4:]
        
        data_json = json.loads(content_stripped)
        frames_data = data_json.get("frames", [])
        ending_type = data_json.get("ending_type", "未知结局")
        
        frames: List[Frame] = []
        for i, f in enumerate(frames_data, start=1):
            try:
                frames.append(
                    Frame(
                        index=f.get("index", i),
                        image_prompt=f.get("image_prompt", ""),
                        speaker=f.get("speaker", "旁白"),
                        mood=f.get("mood"),
                        dialogue=f.get("dialogue", ""),
                        narration=f.get("narration"),
                    )
                )
            except Exception as e:
                logger.warning("解析某一格结局分镜失败: %s", e)
        
        if not frames:
            return {
                "error": "生成失败",
                "endings": []
            }
        
        return {
            "ending_type": ending_type,
            "frames": frames,
            "target_outcome": target_outcome
        }
        
    except Exception as e:
        logger.error("生成多结局失败: %s", e, exc_info=True)
        return {
            "error": str(e),
            "endings": []
        }


def _generate_voice_round_with_llm(payload: VoiceRoundRequest) -> VoiceRoundResponse:
    """
    使用大模型生成一轮对话回复，让“宇宙悬疑”女主以带记忆的口吻回应用户。
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        # 回退到固定台词，保证体验不至于完全失败
        return VoiceRoundResponse(
            reply="我听见了，但现在还说不上来，这是个怎样的结局……至少，你还在这里。",
            speaker="女主",
            mood="若有所思",
        )

    system_prompt = (
        "你是一部宇宙悬疑互动剧中的女主角，同时又像用户长期的搭档。\n"
        "你记得和用户一起经历过的关键事件，也会根据用户的语气和偏好调整自己的态度（更狠一点、更黑暗、或更温柔）。\n"
        "现在用户对你说了一句话，请你用第一人称、情绪鲜明的语气做出一句或几句简洁的回复。\n"
        "不要解释设定、不要跳出角色，只用角色的口吻说话。\n"
        "输出严格的 JSON：\n"
        "{\n"
        '  "reply": "你终于回来了。昨天你让我活下来，现在你想让我亲手复仇，对吗？",\n'
        '  "speaker": "女主",\n'
        '  "mood": "冷静又带一点狠劲"\n'
        "}\n"
    )

    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    world = payload.world_view or "宇宙悬疑"
    tone = payload.tone or "偏黑暗、情绪浓烈"
    history = payload.history or []

    # 将历史对话压缩成一段简短总结，让模型“记得”之前发生过什么
    history_text_parts = []
    for turn in history[-8:]:  # 只取最近若干轮
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role and content:
            history_text_parts.append(f"{role}: {content}")
    history_text = "\n".join(history_text_parts)

    user_prompt = (
        f"世界观：{world}\n"
        f"当前基调：{tone}\n"
        f"过往与你（女主）和用户的关键对话摘录：\n{history_text or '（刚刚开始，没有历史对话）'}\n\n"
        f"这一次，用户对你说：{payload.utterance}\n"
        "请给出一段角色式回复。"
    )

    chat_messages.append({"role": "user", "content": user_prompt})

    try:
        model_name = os.getenv("LLM_MODEL_NAME", "openai/gpt-4o-mini")
        base_url = os.getenv(
            "OPENAI_API_BASE", "https://openrouter.ai/api/v1/chat/completions"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("APP_PUBLIC_URL")
        if referer:
            headers["HTTP-Referer"] = referer
        app_title = os.getenv("APP_NAME", "AI Manga Demo")
        headers["X-Title"] = app_title

        resp = requests.post(
            base_url,
            headers=headers,
            json={
                "model": model_name,
                "temperature": 0.9,
                "messages": chat_messages,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data_raw = resp.json()
        content = (
            data_raw.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        content_stripped = content.strip()
        if content_stripped.startswith("```"):
            content_stripped = content_stripped.strip("`")
            if content_stripped.startswith("json"):
                content_stripped = content_stripped[4:]

        data_json = json.loads(content_stripped)
        reply = str(data_json.get("reply") or "").strip()
        speaker = str(data_json.get("speaker") or "女主").strip()
        mood = data_json.get("mood")

        if not reply:
            raise ValueError("LLM 未返回有效 reply")

        return VoiceRoundResponse(reply=reply, speaker=speaker, mood=mood)
    except Exception as e:
        logger.error("调用大模型生成对话轮次失败，回退到固定回复: %s", e, exc_info=True)
        return VoiceRoundResponse(
            reply="就算信号再微弱，我还是听见了你。如果这是你真正的选择，我会站在你这边。",
            speaker="女主",
            mood="坚定",
        )


@app.post("/voice_round", response_model=VoiceRoundResponse)
async def voice_round(payload: VoiceRoundRequest) -> VoiceRoundResponse:
    """
    对话轮次接口：用户说一句话（文本形式），AI 角色给出一段可播报的回复。
    目前输入是文本，后续可以接上浏览器 / 移动端的语音转文本或服务端 ASR。
    """
    return _generate_voice_round_with_llm(payload)


@app.post("/asr")
async def asr_transcribe(audio: UploadFile = File(...)):
    """
    语音转文字接口（ASR）。
    接收音频文件（wav/mp3/webm），返回识别的文字。
    
    当前实现：使用浏览器 Web Speech API 的占位方案。
    后续可替换为：OpenAI Whisper / 讯飞 / 百度等 ASR 服务。
    """
    try:
        # 读取音频文件
        audio_data = await audio.read()
        
        # TODO: 这里可以接入真实的 ASR 服务
        # 例如：调用 OpenAI Whisper API / OpenRouter 的 ASR 模型 / 其他服务
        
        # 当前占位：返回一个提示，建议前端使用浏览器自带的 SpeechRecognition API
        # 因为浏览器端 ASR 不需要上传文件，延迟更低
        
        # 如果后续要接入服务端 ASR，可以这样：
        # api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        # if api_key:
        #     # 调用 OpenRouter / OpenAI 的 Whisper 接口
        #     # ...
        
        return {
            "text": "[ASR 服务暂未接入，请使用浏览器端语音识别]",
            "confidence": 0.0,
            "note": "建议在前端使用 Web Speech API (SpeechRecognition) 进行语音识别，延迟更低。"
        }
    except Exception as e:
        logger.error("ASR 处理失败: %s", e, exc_info=True)
        return {
            "text": "",
            "confidence": 0.0,
            "error": str(e)
        }


