import asyncio
import hashlib
import json
import math
import re
import time
import datetime
import unicodedata
import weakref
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Dict, Sequence

import astrbot.api.star as star
from astrbot.api.star import StarTools
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api import logger
from astrbot.api.message_components import Plain


TRUNCATION_MARKER = "[消息过长，已截断]"
REPEAT_MARKER = "…[重复已压缩]"
DEFAULT_WEIGHTS = {
    "relevance": 0.25,
    "willingness": 0.2,
    "social": 0.2,
    "timing": 0.15,
    "continuity": 0.2,
}


@dataclass
class JudgeResult:
    """判断结果数据类"""
    relevance: float = 0.0
    willingness: float = 0.0
    social: float = 0.0
    timing: float = 0.0
    continuity: float = 0.0  # 新增：与上次回复的连贯性
    reasoning: str = ""
    should_reply: bool = False
    overall_score: float = 0.0


@dataclass
class RawMessage:
    """原始群聊消息条目"""
    sender_name: str
    sender_id: str
    content: str
    timestamp: float
    is_bot: bool = False
    was_truncated: bool = False
    was_sanitized: bool = False


@dataclass
class ChatState:
    """群聊状态数据类"""
    energy: float = 1.0
    last_reply_time: float = 0.0
    last_energy_update_time: float = 0.0
    last_activity_time: float = 0.0
    last_reset_date: str = ""
    total_messages: int = 0
    total_replies: int = 0


@dataclass(frozen=True)
class HistoryContext:
    text: str
    message_count: int
    chars: int
    truncated_count: int
    sanitized_count: int


@dataclass(frozen=True)
class JudgePromptMetrics:
    history_count: int
    history_chars: int
    current_chars: int
    persona_chars: int
    prompt_chars: int
    truncated_count: int
    sanitized_count: int


def _sanitize_message(text: str, max_combining_marks: int = 2, max_repeats: int = 20) -> tuple[str, bool]:
    """保守清理控制字符、Zalgo 组合符和异常连续重复字符。"""
    output = []
    combining_run = 0
    repeat_char = ""
    repeat_count = 0
    repeat_marker_added = False
    changed = False

    for char in str(text or ""):
        if unicodedata.category(char) in {"Cc", "Cs"} and char not in "\n\t":
            changed = True
            continue

        if unicodedata.combining(char):
            combining_run += 1
            if combining_run > max_combining_marks:
                changed = True
                continue
        else:
            combining_run = 0

        if char == repeat_char:
            repeat_count += 1
            if repeat_count > max_repeats:
                changed = True
                if not repeat_marker_added:
                    output.append(REPEAT_MARKER)
                    repeat_marker_added = True
                continue
        else:
            repeat_char = char
            repeat_count = 1
            repeat_marker_added = False

        output.append(char)

    return "".join(output), changed


def _truncate_message(text: str, max_chars: int) -> tuple[str, bool]:
    """在固定字符预算内保留消息首尾。"""
    if len(text) <= max_chars:
        return text, False
    if max_chars <= len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[:max_chars], True

    remaining = max_chars - len(TRUNCATION_MARKER)
    head_chars = (remaining + 1) // 2
    tail_chars = remaining - head_chars
    tail = text[-tail_chars:] if tail_chars else ""
    return f"{text[:head_chars]}{TRUNCATION_MARKER}{tail}", True


def _format_raw_message(message: RawMessage) -> str:
    prefix = "[机器人]" if message.is_bot else f"[{message.sender_name}]"
    return f"{prefix}: {message.content}"


def _build_bounded_history(
    messages: Sequence[RawMessage], max_messages: int, max_chars: int
) -> HistoryContext:
    """从最新消息向前填充历史，确保格式化结果不超过总预算。"""
    selected = []
    used_chars = 0
    for message in reversed(messages[-max_messages:]):
        line = _format_raw_message(message)
        added_chars = len(line) + (1 if selected else 0)
        if used_chars + added_chars > max_chars:
            break
        selected.append((message, line))
        used_chars += added_chars

    selected.reverse()
    text = "\n".join(line for _, line in selected) or "暂无对话历史"
    return HistoryContext(
        text=text,
        message_count=len(selected),
        chars=used_chars,
        truncated_count=sum(message.was_truncated for message, _ in selected),
        sanitized_count=sum(message.was_sanitized for message, _ in selected),
    )


def _extract_json(text: str) -> dict:
    """从模型返回的文本中稳健地提取 JSON 对象。

    依次尝试：
    1. 直接解析
    2. 去除 markdown 代码块后解析
    3. 正则提取第一个 {...} 子串后解析
    """
    if not isinstance(text, str):
        raise ValueError("模型返回内容不是文本")
    text = text.strip()

    # 1. 直接尝试
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # 2. 去除 markdown 代码块
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # 3. 从每个左花括号开始尝试 raw_decode，避免贪婪正则吞掉多个对象
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start():])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue

    raise ValueError("无法从文本中提取有效 JSON")


def _clamp_score(v) -> float:
    """将模型返回的分数值钉位到 [0, 10]。"""
    try:
        value = float(v)
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(10.0, value))
    except (TypeError, ValueError):
        return 0.0


class HeartflowPlugin(star.Star):

    _PROMPT_CACHE_LIMIT = 128
    _SUMMARY_FAILURE_RETRY_SECONDS = 10 * 60
    _PENDING_REPLY_TIMEOUT_SECONDS = 3 * 60
    _CHAT_STATE_TTL_SECONDS = 7 * 24 * 60 * 60
    _CLEANUP_INTERVAL_SECONDS = 60 * 60

    def __init__(self, context: star.Context, config):
        super().__init__(context)
        self.config = config

        # 判断模型配置
        self.judge_provider_name = self.config.get("judge_provider_name", "")

        # 心流参数配置
        self.reply_threshold = self._config_float("reply_threshold", 0.6, 0.0, 1.0)
        self.energy_decay_rate = self._config_float("energy_decay_rate", 0.1, 0.0, 1.0)
        self.energy_recovery_rate = self._config_float("energy_recovery_rate", 0.02, 0.0, 1.0)
        self.context_messages_count = self._config_int("context_messages_count", 10, 1, 50)
        self.max_message_chars = self._config_int("max_message_chars", 1200, 100, 10000)
        self.max_context_chars = self._config_int("max_context_chars", 6000, 500, 50000)
        if self.max_message_chars > self.max_context_chars:
            logger.warning(
                "max_message_chars 大于 max_context_chars，已按动态上下文预算下调"
            )
            self.max_message_chars = self.max_context_chars
        self.min_reply_interval = self._config_int("min_reply_interval_seconds", 0, 0, 86400)
        self.whitelist_enabled = self.config.get("whitelist_enabled", False)
        self.chat_whitelist = self.config.get("chat_whitelist", [])

        # 群聊状态管理
        self.chat_states: Dict[str, ChatState] = {}
        self._judging_chats: set[str] = set()
        self._pending_reply_chats: Dict[str, float] = {}
        self._last_cleanup_time = 0.0

        # 原始群聊消息缓冲区：{unified_msg_origin: deque[RawMessage]}
        # 记录所有群聊原始消息（无论是否触发 LLM），用于判断上下文
        self._raw_msg_buffer: Dict[str, deque] = {}
        self._raw_msg_buffer_size = self.context_messages_count * 4

        # 人格摘要缓存：persona_id + prompt hash，成功摘要才会写入。
        self.system_prompt_cache: OrderedDict[str, Dict[str, object]] = OrderedDict()
        self._summary_retry_after: OrderedDict[str, float] = OrderedDict()
        self._persona_locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        try:
            self._prompt_cache_file = StarTools.get_data_dir("astrbot_plugin_Heartflow") / "persona_summary_cache.json"
        except Exception as e:
            logger.warning(f"人格摘要缓存目录不可用，将仅使用内存缓存: {e}")
            self._prompt_cache_file = None
        self._load_prompt_cache()

        # 判断配置
        self.judge_include_reasoning = bool(self.config.get("judge_include_reasoning", False))
        self.judge_max_retries = self._config_int("judge_max_retries", 1, 0, 3)
        
        # 判断权重配置
        self.weights = {
            name: self._config_float(f"judge_{name}", default, 0.0, 1.0)
            for name, default in DEFAULT_WEIGHTS.items()
        }
        weight_sum = sum(self.weights.values())
        if weight_sum <= 0:
            logger.warning("判断权重总和为 0，已回退到默认权重")
            self.weights = DEFAULT_WEIGHTS.copy()
        elif abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"判断权重和不为1，当前和为{weight_sum}")
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}

        logger.info("心流插件已初始化")

    def _config_float(self, name: str, default: float, minimum: float, maximum: float) -> float:
        raw = self.config.get(name, default)
        try:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
        except (TypeError, ValueError):
            logger.warning(f"配置 {name}={raw!r} 无效，使用默认值 {default}")
            return default
        clamped = max(minimum, min(maximum, value))
        if clamped != value:
            logger.warning(f"配置 {name}={raw!r} 超出范围，已钳制为 {clamped}")
        return clamped

    def _config_int(self, name: str, default: int, minimum: int, maximum: int) -> int:
        raw = self.config.get(name, default)
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError):
            logger.warning(f"配置 {name}={raw!r} 无效，使用默认值 {default}")
            return default
        clamped = max(minimum, min(maximum, value))
        if clamped != value:
            logger.warning(f"配置 {name}={raw!r} 超出范围，已钳制为 {clamped}")
        return clamped

    @staticmethod
    def _persona_cache_key(persona_id: str, original_prompt: str) -> str:
        prompt_hash = hashlib.sha256(original_prompt.encode("utf-8")).hexdigest()[:20]
        return f"{persona_id}:{prompt_hash}"

    def _summary_backoff_active(self, cache_key: str) -> bool:
        retry_after = self._summary_retry_after.get(cache_key)
        if retry_after is None:
            return False
        if retry_after > time.time():
            self._summary_retry_after.move_to_end(cache_key)
            return True
        self._summary_retry_after.pop(cache_key, None)
        return False

    def _load_prompt_cache(self) -> None:
        if not self._prompt_cache_file or not self._prompt_cache_file.exists():
            return
        try:
            data = json.loads(self._prompt_cache_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("缓存根节点不是对象")
            for key, value in list(data.items())[-self._PROMPT_CACHE_LIMIT:]:
                if isinstance(value, dict) and isinstance(value.get("summarized"), str):
                    self.system_prompt_cache[str(key)] = value
        except Exception as e:
            logger.warning(f"读取人格摘要缓存失败，已忽略: {e}")

    def _save_prompt_cache(self) -> None:
        if not self._prompt_cache_file:
            return
        temp_file = self._prompt_cache_file.with_suffix(".tmp")
        try:
            temp_file.write_text(
                json.dumps(self.system_prompt_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_file.replace(self._prompt_cache_file)
        except Exception as e:
            logger.warning(f"保存人格摘要缓存失败: {e}")
            try:
                temp_file.unlink(missing_ok=True)
            except OSError:
                pass

    async def _get_or_create_summarized_system_prompt(self, event: AstrMessageEvent, original_prompt: str) -> str:
        """获取或创建精简版系统提示词"""
        if not original_prompt or len(original_prompt.strip()) < 50:
            return original_prompt

        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            conversation = None
            if curr_cid:
                conversation = await self.context.conversation_manager.get_conversation(
                    event.unified_msg_origin, curr_cid
                )
            persona_id = (conversation.persona_id if conversation else None) or "default"
        except Exception as e:
            logger.debug(f"获取人格 ID 失败，使用 default 缓存键 ({type(e).__name__})")
            persona_id = "default"

        cache_key = self._persona_cache_key(persona_id, original_prompt)
        cached = self.system_prompt_cache.get(cache_key)
        if cached:
            self.system_prompt_cache.move_to_end(cache_key)
            return str(cached["summarized"])
        if self._summary_backoff_active(cache_key):
            return original_prompt

        lock = self._persona_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self.system_prompt_cache.get(cache_key)
            if cached:
                self.system_prompt_cache.move_to_end(cache_key)
                return str(cached["summarized"])
            if self._summary_backoff_active(cache_key):
                return original_prompt

            summarized_prompt = await self._summarize_system_prompt(original_prompt)
            if not summarized_prompt:
                self._summary_retry_after[cache_key] = (
                    time.time() + self._SUMMARY_FAILURE_RETRY_SECONDS
                )
                self._summary_retry_after.move_to_end(cache_key)
                while len(self._summary_retry_after) > self._PROMPT_CACHE_LIMIT:
                    self._summary_retry_after.popitem(last=False)
                return original_prompt

            self._summary_retry_after.pop(cache_key, None)
            # 同一人格只保留当前 prompt 版本，避免旧内容长期占用缓存。
            stale_keys = [
                key
                for key, value in self.system_prompt_cache.items()
                if value.get("persona_id") == persona_id and key != cache_key
            ]
            for key in stale_keys:
                self.system_prompt_cache.pop(key, None)
                self._persona_locks.pop(key, None)

            self.system_prompt_cache[cache_key] = {
                "persona_id": persona_id,
                "original_chars": len(original_prompt),
                "summarized": summarized_prompt,
            }
            while len(self.system_prompt_cache) > self._PROMPT_CACHE_LIMIT:
                old_key, _ = self.system_prompt_cache.popitem(last=False)
                self._persona_locks.pop(old_key, None)
            self._save_prompt_cache()
            logger.info(
                f"创建人格摘要: [{persona_id}] | 原长度:{len(original_prompt)} -> 摘要长度:{len(summarized_prompt)}"
            )
            return summarized_prompt

    async def _summarize_system_prompt(self, original_prompt: str) -> str | None:
        """使用小模型对系统提示词进行总结"""
        try:
            if not self.judge_provider_name:
                return None
            
            judge_provider = self.context.get_provider_by_id(self.judge_provider_name)
            if not judge_provider:
                return None
            persona_source, persona_truncated = _truncate_message(original_prompt, 12000)
            if persona_truncated:
                logger.debug("人格原文超过 12000 字符，摘要请求已保留首尾并截断")

            summarize_prompt = f"""请将以下机器人角色设定总结为简洁的核心要点，保留关键的性格特征、行为方式和角色定位。
总结后的内容应该在100-200字以内，突出最重要的角色特点。

原始角色设定：
{persona_source}

请以JSON格式回复：
{{
    "summarized_persona": "精简后的角色设定，保留核心特征和行为方式"
}}

**重要：你的回复必须是完整的JSON对象，不要包含任何其他内容！**"""

            llm_response = await judge_provider.text_chat(
                prompt=summarize_prompt,
                contexts=[]  # 不需要上下文
            )

            content = llm_response.completion_text.strip()
            
            # 尝试提取JSON
            try:
                result_data = _extract_json(content)
                summarized = result_data.get("summarized_persona", "")

                if summarized and len(summarized.strip()) > 10:
                    return summarized.strip()
                else:
                    logger.warning("小模型返回的总结内容为空或过短")
                    return None

            except (json.JSONDecodeError, ValueError):
                logger.error(f"小模型总结系统提示词返回非有效JSON，返回长度: {len(content)}")
                return None
                
        except Exception as e:
            logger.error(f"总结系统提示词异常 ({type(e).__name__})")
            return None

    def _build_judge_prompt(
        self,
        event: AstrMessageEvent,
        snapshot: Sequence[RawMessage],
        persona_system_prompt: str,
        chat_state: ChatState,
    ) -> tuple[str, JudgePromptMetrics]:
        current = snapshot[-1]
        history_budget = max(0, self.max_context_chars - len(current.content))
        history = _build_bounded_history(
            snapshot[:-1], self.context_messages_count, history_budget
        )
        persona_for_prompt, _ = _truncate_message(
            persona_system_prompt, min(2000, self.max_context_chars)
        )
        chat_context = self._build_chat_context_from_snapshot(snapshot[:-1], chat_state)
        reasoning_field = (
            ',\n  "reasoning": "简短说明判断依据"'
            if self.judge_include_reasoning
            else ""
        )
        prompt = f"""你是群聊回复决策系统。你的唯一任务是评估机器人是否应主动参与群聊，不要代替机器人回复用户。

固定评分规则：
1. relevance 内容相关度：内容是否有趣、有价值、值得回应（0-10）。
2. willingness 回复意愿：结合角色与精力，是否愿意参与（0-10）。
3. social 社交适宜性：此时加入是否符合群聊氛围（0-10）。
4. timing 时机恰当性：结合回复间隔与消息时效性（0-10）。
5. continuity 对话连贯性：是否承接机器人上次发言；无上次发言时给 5 分（0-10）。

只返回以下 JSON 对象，所有分数必须是 0 到 10 的数字：
{{
  "relevance": 0,
  "willingness": 0,
  "social": 0,
  "timing": 0,
  "continuity": 0{reasoning_field}
}}

机器人角色摘要：
{persona_for_prompt or "默认角色：智能助手"}

本轮动态状态：
- 当前精力：{chat_state.energy:.2f}/1.0
- 距离上次主动回复：{self._get_minutes_since_last_reply(event.unified_msg_origin)} 分钟
- Python 回复阈值：{self.reply_threshold:.2f}
{chat_context}

最近群聊历史（由旧到新）：
{history.text}

当前待判断消息：
- 发送者：{current.sender_name}
- 内容：{current.content}
- 时间：{datetime.datetime.now().strftime('%H:%M:%S')}
"""
        metrics = JudgePromptMetrics(
            history_count=history.message_count,
            history_chars=history.chars,
            current_chars=len(current.content),
            persona_chars=len(persona_for_prompt),
            prompt_chars=len(prompt),
            truncated_count=history.truncated_count + int(current.was_truncated),
            sanitized_count=history.sanitized_count + int(current.was_sanitized),
        )
        return prompt, metrics

    async def judge_with_tiny_model(
        self, event: AstrMessageEvent, snapshot: Sequence[RawMessage]
    ) -> JudgeResult:
        """使用小模型进行智能判断"""

        if not self.judge_provider_name:
            logger.warning("小参数判断模型提供商名称未配置，跳过心流判断")
            return JudgeResult(should_reply=False, reasoning="提供商未配置")

        # 获取指定的 provider
        try:
            judge_provider = self.context.get_provider_by_id(self.judge_provider_name)
            if not judge_provider:
                logger.warning(f"未找到提供商: {self.judge_provider_name}")
                return JudgeResult(should_reply=False, reasoning=f"提供商不存在: {self.judge_provider_name}")
        except Exception as e:
            logger.error(f"获取提供商失败 ({type(e).__name__})")
            return JudgeResult(should_reply=False, reasoning="获取提供商失败")

        # 获取群聊状态
        chat_state = self._get_chat_state(event.unified_msg_origin)

        # 获取当前对话的人格系统提示词，让模型了解大参数LLM的角色设定
        original_persona_prompt = await self._get_persona_system_prompt(event)
        logger.debug(f"小参数模型获取原始人格提示词: {'有' if original_persona_prompt else '无'} | 长度: {len(original_persona_prompt) if original_persona_prompt else 0}")
        
        # 获取或创建精简版系统提示词
        persona_system_prompt = await self._get_or_create_summarized_system_prompt(event, original_persona_prompt)
        logger.debug(f"小参数模型使用精简人格提示词: {'有' if persona_system_prompt else '无'} | 长度: {len(persona_system_prompt) if persona_system_prompt else 0}")

        try:
            complete_judge_prompt, metrics = self._build_judge_prompt(
                event, snapshot, persona_system_prompt, chat_state
            )
            logger.debug(
                "[HeartflowJudge] "
                f"history={metrics.history_count} history_chars={metrics.history_chars} "
                f"current_chars={metrics.current_chars} persona_chars={metrics.persona_chars} "
                f"prompt_chars={metrics.prompt_chars} truncated={metrics.truncated_count} "
                f"sanitized={metrics.sanitized_count}"
            )

            # 重试机制：使用配置的重试次数
            max_retries = self.judge_max_retries + 1
            if self.judge_max_retries == 0:
                max_retries = 1

            for attempt in range(max_retries):
                try:
                    llm_response = await judge_provider.text_chat(
                        prompt=complete_judge_prompt,
                        contexts=[],
                        image_urls=[],
                    )

                    content = llm_response.completion_text.strip()
                    logger.debug(f"小参数模型返回长度: {len(content)}")

                    judge_data = _extract_json(content)

                    # 直接从 JSON 根对象获取分数，并钉位到 [0, 10]
                    relevance = _clamp_score(judge_data.get("relevance", 0))
                    willingness = _clamp_score(judge_data.get("willingness", 0))
                    social = _clamp_score(judge_data.get("social", 0))
                    timing = _clamp_score(judge_data.get("timing", 0))
                    continuity = _clamp_score(judge_data.get("continuity", 0))
                    
                    # 计算综合评分
                    overall_score = (
                        relevance * self.weights["relevance"] +
                        willingness * self.weights["willingness"] +
                        social * self.weights["social"] +
                        timing * self.weights["timing"] +
                        continuity * self.weights["continuity"]
                    ) / 10.0

                    # 根据综合评分判断是否应该回复
                    should_reply = overall_score >= self.reply_threshold

                    logger.debug(f"小参数模型判断成功，综合评分: {overall_score:.3f}, 是否回复: {should_reply}")

                    return JudgeResult(
                        relevance=relevance,
                        willingness=willingness,
                        social=social,
                        timing=timing,
                        continuity=continuity,
                        reasoning=judge_data.get("reasoning", "") if self.judge_include_reasoning else "",
                        should_reply=should_reply,
                        overall_score=overall_score,
                    )
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        f"小参数模型返回JSON解析失败 "
                        f"(尝试 {attempt + 1}/{max_retries}, 错误类型 {type(e).__name__})"
                    )
                    logger.warning(f"无法解析的模型返回长度: {len(content)}")
                    
                    if attempt == max_retries - 1:
                        # 最后一次尝试失败，返回失败结果
                        logger.error(f"小参数模型重试{self.judge_max_retries}次后仍然返回无效JSON，放弃处理")
                        return JudgeResult(should_reply=False, reasoning=f"JSON解析失败，重试{self.judge_max_retries}次")
                    continue

        except Exception as e:
            logger.error(f"小参数模型判断异常 ({type(e).__name__})")
            return JudgeResult(should_reply=False, reasoning="判断异常")

    def _append_raw_message(
        self,
        umo: str,
        sender_name: str,
        sender_id: str,
        content: str,
        is_bot: bool,
    ) -> RawMessage:
        sanitized, was_sanitized = _sanitize_message(content)
        prepared, was_truncated = _truncate_message(sanitized, self.max_message_chars)
        safe_sender, sender_sanitized = _sanitize_message(sender_name)
        safe_sender, sender_truncated = _truncate_message(safe_sender, 80)
        if umo not in self._raw_msg_buffer:
            self._raw_msg_buffer[umo] = deque(maxlen=self._raw_msg_buffer_size)
        message = RawMessage(
            sender_name=safe_sender,
            sender_id=sender_id,
            content=prepared,
            timestamp=time.time(),
            is_bot=is_bot,
            was_truncated=was_truncated,
            was_sanitized=was_sanitized or sender_sanitized or sender_truncated,
        )
        self._raw_msg_buffer[umo].append(message)
        state = self._get_chat_state(umo)
        state.last_activity_time = message.timestamp
        if not is_bot:
            state.total_messages += 1
        return message

    def _record_raw_message(self, event: AstrMessageEvent, is_bot: bool = False) -> RawMessage:
        """清洗、截断并记录群聊消息。"""
        self._cleanup_idle_chats()
        return self._append_raw_message(
            umo=event.unified_msg_origin,
            sender_name="bot" if is_bot else event.get_sender_name(),
            sender_id="bot" if is_bot else str(event.get_sender_id()),
            content=event.message_str,
            is_bot=is_bot,
        )

    def _snapshot_history(self, umo: str) -> tuple[RawMessage, ...]:
        return tuple(self._raw_msg_buffer.get(umo, ()))

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1000)
    async def on_group_message(self, event: AstrMessageEvent):
        """群聊消息处理入口"""
        if not self._should_record_message(event):
            return

        self._record_raw_message(event)
        snapshot = self._snapshot_history(event.unified_msg_origin)
        if not self._should_judge_message(event):
            return

        chat_id = event.unified_msg_origin
        if chat_id in self._judging_chats:
            logger.debug(f"群聊已有 Heartflow 判断请求，仅记录本条消息: {chat_id[:20]}...")
            return

        self._judging_chats.add(chat_id)
        try:
            judge_result = await self.judge_with_tiny_model(event, snapshot)

            if judge_result.should_reply:
                logger.info(f"🔥 心流触发主动回复 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f}")

                pending_since = time.time()
                self._pending_reply_chats[chat_id] = pending_since
                event.is_at_or_wake_command = True
                event.set_extra("heartflow_triggered", True)
                event.set_extra("heartflow_pending", True)
                event.set_extra("heartflow_pending_since", pending_since)
                logger.info(f"💖 心流设置唤醒标志 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f}")
                return

            logger.debug(f"心流判断不通过 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f}")

        except Exception as e:
            logger.error(f"心流插件处理消息异常 ({type(e).__name__})")
        finally:
            self._judging_chats.discard(chat_id)

    @filter.after_message_sent()
    async def on_after_message_sent(self, event: AstrMessageEvent):
        """记录 Bot 输出，并仅提交已生成的 Heartflow 主动回复状态。"""
        if not self.config.get("enable_heartflow", False):
            return
        if event.is_private_chat():
            return

        result = event.get_result()
        is_pending_heartflow = (
            event.get_extra("heartflow_triggered", False)
            and event.get_extra("heartflow_pending", False)
            and not event.get_extra("heartflow_committed", False)
        )

        reply_text = ""
        if result is not None and result.chain:
            reply_text = "".join(
                comp.text for comp in result.chain if isinstance(comp, Plain)
            ).strip()
        if reply_text and not event.get_extra("heartflow_history_recorded", False):
            self._append_raw_message(
                event.unified_msg_origin, "bot", "bot", reply_text, is_bot=True
            )
            event.set_extra("heartflow_history_recorded", True)
            logger.debug(f"机器人回复已写入缓冲区: {event.unified_msg_origin[:20]}...")

        if not is_pending_heartflow:
            return

        if result is not None and result.chain and self._is_llm_result(result):
            self._update_active_state(event)
            event.set_extra("heartflow_committed", True)
        pending_since = event.get_extra("heartflow_pending_since")
        if self._pending_reply_chats.get(event.unified_msg_origin) == pending_since:
            self._pending_reply_chats.pop(event.unified_msg_origin, None)
        event.set_extra("heartflow_pending", False)

    @staticmethod
    def _is_llm_result(result) -> bool:
        method = getattr(result, "is_llm_result", None)
        if not callable(method):
            return False
        try:
            return bool(method())
        except Exception:
            return False

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """心流触发时，在 LLM 请求前注入一条提示，让大模型知道自己是主动参与群聊的"""
        if not event.get_extra("heartflow_triggered"):
            return
        if event.get_extra("heartflow_note_injected", False):
            return
        if not req or not hasattr(req, "system_prompt"):
            return
        note = "（注意：本次是你主动参与群聊的，不是用户叫你。回复应自然随意，像普通群成员一样加入话题。）"
        if note not in (req.system_prompt or ""):
            req.system_prompt = (req.system_prompt or "") + "\n" + note
        event.set_extra("heartflow_note_injected", True)

    def _should_record_message(self, event: AstrMessageEvent) -> bool:
        """检查消息是否属于应记录的群聊事实。"""
        if not self.config.get("enable_heartflow", False):
            return False

        if self.whitelist_enabled:
            if not self.chat_whitelist:
                logger.debug(f"白名单为空，跳过处理: {event.unified_msg_origin}")
                return False

            if event.unified_msg_origin not in self.chat_whitelist:
                logger.debug(f"群聊不在白名单中，跳过处理: {event.unified_msg_origin}")
                return False

        # 跳过机器人自己的消息
        if event.get_sender_id() == event.get_self_id():
            return False

        # 跳过空消息
        if not event.message_str or not event.message_str.strip():
            return False

        return True

    def _should_judge_message(self, event: AstrMessageEvent) -> bool:
        """记录之后判断本条消息是否应调用小模型。"""
        chat_id = event.unified_msg_origin
        pending_since = self._pending_reply_chats.get(chat_id)
        if pending_since is not None:
            if time.time() - pending_since < self._PENDING_REPLY_TIMEOUT_SECONDS:
                logger.debug("群聊已有 Heartflow 主动回复待发送，仅记录历史")
                return False
            self._pending_reply_chats.pop(chat_id, None)
            logger.debug("Heartflow 主动回复 pending 已超时，允许重新判断")

        if event.is_at_or_wake_command:
            logger.debug("消息已被 @ 或其他插件唤醒，仅记录历史")
            return False

        if self.min_reply_interval > 0:
            state = self._get_chat_state(event.unified_msg_origin)
            elapsed_seconds = (
                time.time() - state.last_reply_time if state.last_reply_time else float("inf")
            )
            if elapsed_seconds < self.min_reply_interval:
                logger.debug(f"冷却中，距上次回复还有 {self.min_reply_interval - elapsed_seconds:.0f}s")
                return False

        return True

    def _get_chat_state(self, chat_id: str) -> ChatState:
        """获取群聊状态"""
        now = time.time()
        if chat_id not in self.chat_states:
            self.chat_states[chat_id] = ChatState(
                last_energy_update_time=now,
                last_activity_time=now,
            )

        # 检查日期重置
        today = datetime.date.today().isoformat()
        state = self.chat_states[chat_id]

        if state.last_reset_date != today:
            state.last_reset_date = today
            # 每日重置时恒复一些精力
            state.energy = min(1.0, state.energy + 0.2)

        # energy_recovery_rate 表示每 5 分钟恢复的精力；计时点与回复时间分离。
        if state.last_energy_update_time <= 0:
            state.last_energy_update_time = now
        elapsed_seconds = max(0.0, now - state.last_energy_update_time)
        if elapsed_seconds > 0:
            time_recovery = (elapsed_seconds / 300.0) * self.energy_recovery_rate
            state.energy = min(1.0, state.energy + time_recovery)
            state.last_energy_update_time = now

        return state

    def _get_minutes_since_last_reply(self, chat_id: str) -> int:
        """获取距离上次回复的分钟数"""
        chat_state = self._get_chat_state(chat_id)

        if chat_state.last_reply_time == 0:
            return 999  # 从未回复过

        return max(0, int((time.time() - chat_state.last_reply_time) / 60))

    def _cleanup_idle_chats(self) -> None:
        now = time.time()
        if now - self._last_cleanup_time < self._CLEANUP_INTERVAL_SECONDS:
            return
        self._last_cleanup_time = now
        stale = [
            chat_id
            for chat_id, state in self.chat_states.items()
            if chat_id not in self._judging_chats
            and state.last_activity_time > 0
            and now - state.last_activity_time > self._CHAT_STATE_TTL_SECONDS
        ]
        for chat_id in stale:
            self.chat_states.pop(chat_id, None)
            self._raw_msg_buffer.pop(chat_id, None)
            self._pending_reply_chats.pop(chat_id, None)
        if stale:
            logger.debug(f"已清理 {len(stale)} 个长期空闲 Heartflow 群状态")

    def _build_chat_context_from_snapshot(
        self, history: Sequence[RawMessage], chat_state: ChatState
    ) -> str:
        """基于判断开始时的快照计算时间窗口活跃度与回复后互动。"""
        now = time.time()
        messages_5m = sum(1 for message in history if now - message.timestamp <= 300)
        messages_10m = sum(1 for message in history if now - message.timestamp <= 600)
        if messages_5m >= 10:
            activity_level = "高"
        elif messages_5m >= 4:
            activity_level = "中"
        else:
            activity_level = "低"

        engagement = ""
        for index in range(len(history) - 1, -1, -1):
            if history[index].is_bot:
                user_messages_after = sum(not message.is_bot for message in history[index + 1:])
                if user_messages_after >= 3:
                    engagement = "\n- 上次机器人回复后：讨论活跃"
                elif user_messages_after == 0:
                    engagement = "\n- 上次机器人回复后：暂无用户接话"
                break

        reply_rate = chat_state.total_replies / max(1, chat_state.total_messages) * 100
        return (
            f"- 最近活跃度：{activity_level}（5分钟 {messages_5m} 条，10分钟 {messages_10m} 条）\n"
            f"- Heartflow 历史回复率：{reply_rate:.1f}%{engagement}"
        )

    def _update_active_state(self, event: AstrMessageEvent):
        """在 after_message_sent 确认模型结果后提交主动回复状态。"""
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        now = time.time()
        chat_state.last_reply_time = now
        chat_state.last_energy_update_time = now
        chat_state.last_activity_time = now
        chat_state.total_replies += 1
        chat_state.energy = max(0.1, chat_state.energy - self.energy_decay_rate)

        logger.debug(f"更新主动状态: {chat_id[:20]}... | 精力: {chat_state.energy:.2f}")

    # 管理员命令：查看心流状态
    @filter.command("heartflow")
    async def heartflow_status(self, event: AstrMessageEvent):
        """查看心流状态"""

        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        status_info = f"""
🔮 心流状态报告

📊 **当前状态**
- 群聊ID: {event.unified_msg_origin}
- 精力水平: {chat_state.energy:.2f}/1.0 {'🟢' if chat_state.energy > 0.7 else '🟡' if chat_state.energy > 0.3 else '🔴'}
- 上次回复: {self._get_minutes_since_last_reply(chat_id)}分钟前

📈 **历史统计**
- 总消息数: {chat_state.total_messages}
- 总回复数: {chat_state.total_replies}
- 回复率: {(chat_state.total_replies / max(1, chat_state.total_messages) * 100):.1f}%

⚙️ **配置参数**
- 回复阈值: {self.reply_threshold}
- 判断提供商: {self.judge_provider_name}
- 最大重试次数: {self.judge_max_retries}
- 白名单模式: {'✅ 开启' if self.whitelist_enabled else '❌ 关闭'}
- 白名单群聊数: {len(self.chat_whitelist) if self.whitelist_enabled else 0}

🧠 **智能缓存**
- 系统提示词缓存: {len(self.system_prompt_cache)} 个

🎯 **评分权重**
- 内容相关度: {self.weights['relevance']:.0%}
- 回复意愿: {self.weights['willingness']:.0%}
- 社交适宜性: {self.weights['social']:.0%}
- 时机恰当性: {self.weights['timing']:.0%}
- 对话连贯性: {self.weights['continuity']:.0%}

🎯 **插件状态**: {'✅ 已启用' if self.config.get('enable_heartflow', False) else '❌ 已禁用'}
"""

        event.set_result(event.plain_result(status_info))

    # 管理员命令：重置心流状态
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("heartflow_reset")
    async def heartflow_reset(self, event: AstrMessageEvent):
        """重置心流状态"""

        chat_id = event.unified_msg_origin
        if chat_id in self.chat_states:
            del self.chat_states[chat_id]
        self._pending_reply_chats.pop(chat_id, None)

        event.set_result(event.plain_result("✅ 心流状态已重置"))
        logger.info(f"心流状态已重置: {chat_id}")

    # 管理员命令：查看系统提示词缓存
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("heartflow_cache")
    async def heartflow_cache_status(self, event: AstrMessageEvent):
        """查看系统提示词缓存状态"""
        
        cache_info = "🧠 系统提示词缓存状态\n\n"
        
        if not self.system_prompt_cache:
            cache_info += "📭 当前无缓存记录"
        else:
            cache_info += f"📝 总缓存数量: {len(self.system_prompt_cache)}\n\n"
            
            for cache_key, cache_data in self.system_prompt_cache.items():
                original_len = int(cache_data.get("original_chars", 0))
                summarized_len = len(cache_data.get("summarized", ""))
                persona_id = cache_data.get("persona_id", "unknown")
                
                cache_info += f"🔑 **缓存键**: {cache_key}\n"
                cache_info += f"👤 **人格ID**: {persona_id}\n"
                cache_info += f"📏 **压缩率**: {original_len} -> {summarized_len} ({(1-summarized_len/max(1,original_len))*100:.1f}% 压缩)\n"
                cache_info += f"📄 **精简内容**: {cache_data.get('summarized', '')[:100]}...\n\n"
        
        event.set_result(event.plain_result(cache_info))

    # 管理员命令：清除系统提示词缓存
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("heartflow_cache_clear")
    async def heartflow_cache_clear(self, event: AstrMessageEvent):
        """清除系统提示词缓存"""
        
        cache_count = len(self.system_prompt_cache)
        self.system_prompt_cache.clear()
        self._summary_retry_after.clear()
        self._persona_locks.clear()
        self._save_prompt_cache()
        
        event.set_result(event.plain_result(f"✅ 已清除 {cache_count} 个系统提示词缓存"))
        logger.info(f"系统提示词缓存已清除，共清除 {cache_count} 个缓存")

    async def _get_persona_system_prompt(self, event: AstrMessageEvent) -> str:
        """获取当前对话的人格系统提示词"""
        try:
            persona_mgr = self.context.persona_manager

            # 获取当前对话，尝试拿到会话绑定的 persona_id
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            persona_id: str | None = None
            if curr_cid:
                conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
                if conversation:
                    persona_id = conversation.persona_id

            # 用户显式取消人格
            if persona_id == "[%None]":
                return ""

            if persona_id:
                # 直接通过 PersonaManager 查询数据库
                try:
                    persona = await persona_mgr.get_persona(persona_id)
                    return persona.system_prompt or ""
                except ValueError:
                    logger.debug(f"未找到人格 {persona_id}，回退到默认人格")

            # 无 persona_id 或查询失败，使用默认人格
            default_persona = await persona_mgr.get_default_persona_v3(event.unified_msg_origin)
            return default_persona.get("prompt", "")

        except Exception as e:
            logger.debug(f"获取人格系统提示词失败 ({type(e).__name__})")
            return ""
