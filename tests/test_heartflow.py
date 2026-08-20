import asyncio
import datetime
import importlib
import json
import math
import sys
import tempfile
import types
import unittest
import unicodedata
from pathlib import Path
from unittest.mock import patch


class Plain:
    def __init__(self, text):
        self.text = text


class Result:
    LLM_RESULT = "llm"
    AGENT_RUNNER_ERROR = "agent_runner_error"
    GENERAL_RESULT = "general"

    def __init__(self, chain, content_type=LLM_RESULT):
        self.chain = chain
        self.content_type = content_type

    def is_llm_result(self):
        return self.content_type == self.LLM_RESULT

    def is_model_result(self):
        return self.content_type in {self.LLM_RESULT, self.AGENT_RUNNER_ERROR}


class LLMResponse:
    def __init__(self, text):
        self.completion_text = text


class Event:
    def __init__(
        self, text="hello", session_id="group", sender_id="user", result=None, wake=False, private=False
    ):
        self.message_str = text
        self.unified_msg_origin = session_id
        self._sender_id = sender_id
        self._result = result
        self.is_at_or_wake_command = wake
        self._private = private
        self.extras = {}

    def is_private_chat(self):
        return self._private

    def get_sender_id(self):
        return self._sender_id

    def get_sender_name(self):
        return "Sender"

    def get_self_id(self):
        return "bot"

    def get_result(self):
        return self._result

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def plain_result(self, text):
        return Result([Plain(text)], content_type=Result.GENERAL_RESULT)

    def set_result(self, result):
        self._result = result


class StarTools:
    data_dir = Path(tempfile.gettempdir()) / "heartflow-tests-default"

    @classmethod
    def get_data_dir(cls, _plugin_name=None):
        cls.data_dir.mkdir(parents=True, exist_ok=True)
        return cls.data_dir


def _identity_decorator(*_args, **_kwargs):
    return lambda func: func


def _install_astrbot_stubs():
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event = types.ModuleType("astrbot.api.event")
    star = types.ModuleType("astrbot.api.star")
    components = types.ModuleType("astrbot.api.message_components")

    class Logger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    class Star:
        def __init__(self, context):
            self.context = context

    api.logger = Logger()
    event.AstrMessageEvent = object
    event.filter = types.SimpleNamespace(
        EventMessageType=types.SimpleNamespace(GROUP_MESSAGE="group"),
        PermissionType=types.SimpleNamespace(ADMIN="admin"),
        event_message_type=_identity_decorator,
        after_message_sent=_identity_decorator,
        on_llm_request=_identity_decorator,
        command=_identity_decorator,
        permission_type=_identity_decorator,
    )
    star.Context = object
    star.Star = Star
    star.StarTools = StarTools
    components.Plain = Plain

    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.star": star,
            "astrbot.api.message_components": components,
        }
    )


_install_astrbot_stubs()
sys.modules.pop("main", None)
plugin_module = importlib.import_module("main")


class HeartflowCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        StarTools.data_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_plugin(self, **config):
        defaults = {"enable_heartflow": True}
        defaults.update(config)
        return plugin_module.HeartflowPlugin(context=types.SimpleNamespace(), config=defaults)

    def test_extract_json_accepts_plain_fenced_and_wrapped_objects(self):
        expected = {"relevance": 8}
        self.assertEqual(plugin_module._extract_json('{"relevance": 8}'), expected)
        self.assertEqual(plugin_module._extract_json('```json\n{"relevance": 8}\n```'), expected)
        self.assertEqual(plugin_module._extract_json('result: {"relevance": 8} done'), expected)

    def test_extract_json_rejects_non_object_json(self):
        with self.assertRaises(ValueError):
            plugin_module._extract_json("[]")
        with self.assertRaises(ValueError):
            plugin_module._extract_json(None)

    def test_invalid_json_never_leaks_model_content_to_exception_or_logs(self):
        marker = "SECRET_CONTENT_SHOULD_NOT_APPEAR"
        with self.assertRaises(ValueError) as raised:
            plugin_module._extract_json(marker)
        self.assertNotIn(marker, str(raised.exception))

        class Provider:
            async def text_chat(self, **_kwargs):
                return LLMResponse(marker)

        class ConversationManager:
            async def get_curr_conversation_id(self, _umo):
                return None

        class PersonaManager:
            async def get_default_persona_v3(self, _umo):
                return {"prompt": "short persona"}

        plugin = plugin_module.HeartflowPlugin(
            types.SimpleNamespace(
                get_provider_by_id=lambda _name: Provider(),
                conversation_manager=ConversationManager(),
                persona_manager=PersonaManager(),
            ),
            {
                "enable_heartflow": True,
                "judge_provider_name": "judge",
                "judge_max_retries": 0,
            },
        )
        snapshot = (plugin_module.RawMessage("u", "1", "current", 1),)

        with (
            patch.object(plugin_module.logger, "warning") as warning,
            patch.object(plugin_module.logger, "error") as error,
        ):
            asyncio.run(plugin.judge_with_tiny_model(Event("current"), snapshot))

        logged = " ".join(str(call) for call in warning.call_args_list + error.call_args_list)
        self.assertNotIn(marker, logged)

    def test_clamp_score_handles_bounds_and_invalid_values(self):
        self.assertEqual(plugin_module._clamp_score(-3), 0.0)
        self.assertEqual(plugin_module._clamp_score("7.5"), 7.5)
        self.assertEqual(plugin_module._clamp_score(99), 10.0)
        self.assertEqual(plugin_module._clamp_score(None), 0.0)

    def test_sanitize_message_limits_combining_marks_and_controls(self):
        text = "A" + "\u0301" * 20 + "\x00正常中文🙂"
        cleaned, changed = plugin_module._sanitize_message(text)

        self.assertTrue(changed)
        self.assertNotIn("\x00", cleaned)
        self.assertIn("正常中文🙂", cleaned)
        self.assertLessEqual(sum(1 for char in cleaned if unicodedata.combining(char)), 2)

    def test_truncate_message_keeps_head_tail_within_budget(self):
        text = "a" * 12000 + "END"
        shortened, truncated = plugin_module._truncate_message(text, 1200)

        self.assertTrue(truncated)
        self.assertLessEqual(len(shortened), 1200)
        self.assertTrue(shortened.startswith("a"))
        self.assertTrue(shortened.endswith("END"))
        self.assertIn("[消息过长，已截断]", shortened)

    def test_context_budget_prefers_newest_messages(self):
        messages = [
            plugin_module.RawMessage("u1", "1", "old-" + "a" * 20, 1),
            plugin_module.RawMessage("u2", "2", "middle-" + "b" * 20, 2),
            plugin_module.RawMessage("u3", "3", "newest-" + "c" * 20, 3),
        ]

        context = plugin_module._build_bounded_history(messages, max_messages=3, max_chars=50)

        self.assertLessEqual(context.chars, 50)
        self.assertIn("newest", context.text)
        self.assertNotIn("old-", context.text)

    def test_energy_recovery_does_not_change_last_reply_time(self):
        plugin = self.make_plugin(energy_recovery_rate=0.02)
        today = datetime.date.fromtimestamp(1600.0).isoformat()
        plugin.chat_states["group"] = plugin_module.ChatState(
            energy=0.5,
            last_reply_time=1000.0,
            last_energy_update_time=1000.0,
            last_reset_date=today,
        )

        with patch.object(plugin_module.time, "time", return_value=1600.0):
            state = plugin._get_chat_state("group")
            minutes = plugin._get_minutes_since_last_reply("group")

        self.assertAlmostEqual(state.energy, 0.54)
        self.assertEqual(state.last_reply_time, 1000.0)
        self.assertEqual(state.last_energy_update_time, 1600.0)
        self.assertEqual(minutes, 10)

    def test_invalid_weights_fall_back_to_safe_defaults(self):
        plugin = self.make_plugin(
            reply_threshold=2,
            context_messages_count=0,
            judge_relevance=0,
            judge_willingness=-1,
            judge_social=float("nan"),
            judge_timing=0,
            judge_continuity=0,
        )

        self.assertEqual(plugin.reply_threshold, 1.0)
        self.assertGreaterEqual(plugin.context_messages_count, 1)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in plugin.weights.values()))
        self.assertAlmostEqual(sum(plugin.weights.values()), 1.0)

    def test_single_message_limit_cannot_exceed_dynamic_context_budget(self):
        plugin = self.make_plugin(max_message_chars=10000, max_context_chars=500)

        self.assertEqual(plugin.max_message_chars, 500)

    def test_history_snapshot_is_immutable(self):
        plugin = self.make_plugin()
        event = Event("first")
        plugin._record_raw_message(event)
        snapshot = plugin._snapshot_history("group")

        plugin._record_raw_message(Event("future"))

        self.assertIsInstance(snapshot, tuple)
        self.assertEqual([message.content for message in snapshot], ["first"])

    def test_sender_name_is_sanitized_and_bounded(self):
        plugin = self.make_plugin()
        message = plugin._append_raw_message(
            "group", "N" * 1000 + "\u0301" * 50, "user", "hello", is_bot=False
        )

        self.assertLessEqual(len(message.sender_name), 80)
        self.assertLessEqual(
            sum(1 for char in message.sender_name if unicodedata.combining(char)), 2
        )

    def test_very_long_zalgo_message_is_safe_before_buffering(self):
        plugin = self.make_plugin()
        message = plugin._append_raw_message(
            "group", "Sender", "user", "A" + "\u0301" * 15000 + "TAIL", is_bot=False
        )

        self.assertLessEqual(len(message.content), 1200)
        self.assertLessEqual(
            sum(1 for char in message.content if unicodedata.combining(char)), 2
        )
        self.assertTrue(message.was_sanitized)

    def test_wake_message_is_recorded_without_being_judged(self):
        plugin = self.make_plugin()
        event = Event("@bot hello", wake=True)
        calls = 0

        async def fake_judge(_event, _snapshot):
            nonlocal calls
            calls += 1
            return plugin_module.JudgeResult()

        plugin.judge_with_tiny_model = fake_judge
        asyncio.run(plugin.on_group_message(event))

        self.assertEqual(calls, 0)
        self.assertEqual([message.content for message in plugin._snapshot_history("group")], ["@bot hello"])

    def test_cooldown_message_is_recorded_without_being_judged(self):
        plugin = self.make_plugin(min_reply_interval_seconds=60)
        plugin._get_chat_state("group").last_reply_time = plugin_module.time.time()
        calls = 0

        async def fake_judge(_event, _snapshot):
            nonlocal calls
            calls += 1
            return plugin_module.JudgeResult()

        plugin.judge_with_tiny_model = fake_judge
        asyncio.run(plugin.on_group_message(Event("during cooldown")))

        self.assertEqual(calls, 0)
        self.assertEqual(plugin._snapshot_history("group")[-1].content, "during cooldown")

    def test_negative_judge_keeps_event_passive(self):
        plugin = self.make_plugin()
        event = Event("ordinary chat")

        async def fake_judge(_event, _snapshot):
            return plugin_module.JudgeResult(should_reply=False, overall_score=0.1)

        plugin.judge_with_tiny_model = fake_judge
        asyncio.run(plugin.on_group_message(event))

        self.assertFalse(event.is_at_or_wake_command)
        self.assertEqual(plugin._get_chat_state("group").total_replies, 0)

    def test_inflight_group_records_future_message_without_second_judge(self):
        plugin = self.make_plugin()
        entered = asyncio.Event()
        release = asyncio.Event()
        snapshots = []

        async def fake_judge(_event, snapshot=None):
            snapshots.append(snapshot)
            entered.set()
            await release.wait()
            return plugin_module.JudgeResult()

        plugin.judge_with_tiny_model = fake_judge

        async def scenario():
            first = asyncio.create_task(plugin.on_group_message(Event("first")))
            await entered.wait()
            try:
                await asyncio.wait_for(plugin.on_group_message(Event("future")), timeout=0.2)
            finally:
                release.set()
                await first

        asyncio.run(scenario())

        self.assertEqual(len(snapshots), 1)
        self.assertEqual([message.content for message in snapshots[0]], ["first"])
        self.assertEqual([message.content for message in plugin._snapshot_history("group")], ["first", "future"])

    def test_llm_result_commits_active_state_once_and_clears_pending(self):
        plugin = self.make_plugin()
        event = Event(result=Result([Plain("reply")], Result.LLM_RESULT))
        event.set_extra("heartflow_triggered", True)
        event.set_extra("heartflow_pending", True)
        pending_since = plugin_module.time.time()
        event.set_extra("heartflow_pending_since", pending_since)
        plugin._pending_reply_chats["group"] = pending_since

        asyncio.run(plugin.on_after_message_sent(event))
        asyncio.run(plugin.on_after_message_sent(event))

        state = plugin._get_chat_state("group")
        self.assertEqual(state.total_replies, 1)
        self.assertGreater(state.last_reply_time, 0)
        bot_messages = [message for message in plugin._snapshot_history("group") if message.is_bot]
        self.assertEqual(len(bot_messages), 1)
        self.assertNotIn("group", plugin._pending_reply_chats)

    def test_agent_runner_error_does_not_commit_and_clears_pending(self):
        plugin = self.make_plugin()
        event = Event(result=Result([Plain("error")], Result.AGENT_RUNNER_ERROR))
        event.set_extra("heartflow_triggered", True)
        event.set_extra("heartflow_pending", True)
        pending_since = plugin_module.time.time()
        event.set_extra("heartflow_pending_since", pending_since)
        plugin._pending_reply_chats["group"] = pending_since

        asyncio.run(plugin.on_after_message_sent(event))

        self.assertEqual(plugin._get_chat_state("group").total_replies, 0)
        self.assertNotIn("group", plugin._pending_reply_chats)

    def test_general_result_does_not_commit_and_clears_pending(self):
        plugin = self.make_plugin()
        event = Event(result=Result([Plain("general")], Result.GENERAL_RESULT))
        event.set_extra("heartflow_triggered", True)
        event.set_extra("heartflow_pending", True)
        pending_since = plugin_module.time.time()
        event.set_extra("heartflow_pending_since", pending_since)
        plugin._pending_reply_chats["group"] = pending_since

        asyncio.run(plugin.on_after_message_sent(event))

        self.assertEqual(plugin._get_chat_state("group").total_replies, 0)
        self.assertNotIn("group", plugin._pending_reply_chats)

    def test_legacy_model_result_without_llm_check_fails_closed(self):
        class LegacyResult:
            chain = [Plain("legacy")]

            @staticmethod
            def is_model_result():
                return True

        plugin = self.make_plugin()
        event = Event(result=LegacyResult())
        event.set_extra("heartflow_triggered", True)
        event.set_extra("heartflow_pending", True)

        asyncio.run(plugin.on_after_message_sent(event))

        self.assertEqual(plugin._get_chat_state("group").total_replies, 0)

    def test_ordinary_llm_reply_does_not_commit_active_state(self):
        plugin = self.make_plugin()
        ordinary = Event(text="@bot hi", result=Result([Plain("normal reply")]), wake=True)

        asyncio.run(plugin.on_after_message_sent(ordinary))

        self.assertEqual(plugin._get_chat_state("group").total_replies, 0)
        self.assertEqual(len([m for m in plugin._snapshot_history("group") if m.is_bot]), 1)

    def test_private_bot_reply_is_not_added_to_group_history(self):
        plugin = self.make_plugin()
        event = Event(result=Result([Plain("private reply")]), private=True)

        asyncio.run(plugin.on_after_message_sent(event))

        self.assertEqual(plugin._snapshot_history("group"), ())

    def test_pending_reply_records_new_message_without_second_judge(self):
        plugin = self.make_plugin()
        calls = 0

        async def fake_judge(_event, _snapshot):
            nonlocal calls
            calls += 1
            return plugin_module.JudgeResult(should_reply=True, overall_score=0.9)

        plugin.judge_with_tiny_model = fake_judge
        asyncio.run(plugin.on_group_message(Event("first")))
        asyncio.run(plugin.on_group_message(Event("while pending")))

        self.assertEqual(calls, 1)
        self.assertEqual(
            [message.content for message in plugin._snapshot_history("group")],
            ["first", "while pending"],
        )
        self.assertIn("group", plugin._pending_reply_chats)

    def test_expired_pending_reply_allows_judge_again(self):
        plugin = self.make_plugin()
        calls = 0

        async def fake_judge(_event, _snapshot):
            nonlocal calls
            calls += 1
            return plugin_module.JudgeResult()

        plugin.judge_with_tiny_model = fake_judge
        plugin._pending_reply_chats["group"] = 100.0
        with patch.object(
            plugin_module.time,
            "time",
            return_value=100.0 + plugin._PENDING_REPLY_TIMEOUT_SECONDS + 1,
        ):
            asyncio.run(plugin.on_group_message(Event("after timeout")))

        self.assertEqual(calls, 1)
        self.assertNotIn("group", plugin._pending_reply_chats)

    def test_late_old_reply_does_not_clear_new_pending_reply(self):
        plugin = self.make_plugin()
        old_event = Event(result=Result([Plain("late reply")], Result.LLM_RESULT))
        old_event.set_extra("heartflow_triggered", True)
        old_event.set_extra("heartflow_pending", True)
        old_event.set_extra("heartflow_pending_since", 100.0)
        plugin._pending_reply_chats["group"] = 100.0

        async def approve(_event, _snapshot):
            return plugin_module.JudgeResult(should_reply=True, overall_score=0.9)

        plugin.judge_with_tiny_model = approve
        new_pending_since = 100.0 + plugin._PENDING_REPLY_TIMEOUT_SECONDS + 1
        with patch.object(plugin_module.time, "time", return_value=new_pending_since):
            asyncio.run(plugin.on_group_message(Event("new trigger")))
            asyncio.run(plugin.on_after_message_sent(old_event))

        self.assertEqual(plugin._pending_reply_chats["group"], new_pending_since)

    def test_prompt_contains_persona_and_history_once_and_uses_empty_contexts(self):
        plugin = self.make_plugin()
        snapshot = (
            plugin_module.RawMessage("u", "1", "history-marker", 1),
            plugin_module.RawMessage("u", "1", "current-marker", 2),
        )

        prompt, metrics = plugin._build_judge_prompt(
            Event("current-marker"), snapshot, "persona-marker", plugin._get_chat_state("group")
        )

        self.assertEqual(prompt.count("persona-marker"), 1)
        self.assertEqual(prompt.count("history-marker"), 1)
        self.assertEqual(prompt.count("current-marker"), 1)
        self.assertLessEqual(metrics.history_chars, plugin.max_context_chars)

    def test_judge_provider_receives_empty_contexts_and_stable_retry_prompt(self):
        class Provider:
            def __init__(self):
                self.calls = []

            async def text_chat(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return LLMResponse("not json")
                return LLMResponse(
                    '{"relevance":5,"willingness":5,"social":5,"timing":5,"continuity":5}'
                )

        provider = Provider()

        class ConversationManager:
            async def get_curr_conversation_id(self, _umo):
                return None

        class PersonaManager:
            async def get_default_persona_v3(self, _umo):
                return {"prompt": "short persona"}

        context = types.SimpleNamespace(
            get_provider_by_id=lambda _name: provider,
            conversation_manager=ConversationManager(),
            persona_manager=PersonaManager(),
        )
        plugin = plugin_module.HeartflowPlugin(
            context, {"enable_heartflow": True, "judge_provider_name": "judge", "judge_max_retries": 1}
        )
        snapshot = (plugin_module.RawMessage("u", "1", "current-marker", 1),)

        asyncio.run(plugin.judge_with_tiny_model(Event("current-marker"), snapshot))

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[0]["contexts"], [])
        self.assertEqual(provider.calls[0]["prompt"], provider.calls[1]["prompt"])
        self.assertEqual(provider.calls[0]["prompt"].count("current-marker"), 1)

    def test_dynamic_context_budget_includes_current_message(self):
        plugin = self.make_plugin(max_context_chars=1500, max_message_chars=1200)
        snapshot = tuple(
            plugin_module.RawMessage(f"u{i}", str(i), char * 1200, i)
            for i, char in enumerate("abcd", start=1)
        )

        _, metrics = plugin._build_judge_prompt(
            Event("d" * 1200), snapshot, "persona", plugin._get_chat_state("group")
        )

        self.assertLessEqual(metrics.history_chars + metrics.current_chars, 1500)

    def test_failed_persona_summary_fallback_is_bounded_in_judge_prompt(self):
        plugin = self.make_plugin()
        snapshot = (plugin_module.RawMessage("u", "1", "current", 1),)

        prompt, metrics = plugin._build_judge_prompt(
            Event("current"), snapshot, "P" * 5000, plugin._get_chat_state("group")
        )

        self.assertLessEqual(metrics.persona_chars, 2000)
        self.assertIn("[消息过长，已截断]", prompt)

    def test_persona_cache_survives_restart_and_prompt_hash_changes_key(self):
        first = self.make_plugin()
        key_one = first._persona_cache_key("persona", "prompt one")
        key_two = first._persona_cache_key("persona", "prompt two")
        first.system_prompt_cache[key_one] = {
            "persona_id": "persona",
            "original_chars": 10,
            "summarized": "summary",
        }
        first._save_prompt_cache()

        second = self.make_plugin()

        self.assertNotEqual(key_one, key_two)
        self.assertEqual(second.system_prompt_cache[key_one]["summarized"], "summary")

    def test_concurrent_persona_requests_share_one_summary_call(self):
        class ConversationManager:
            async def get_curr_conversation_id(self, _umo):
                return None

        plugin = plugin_module.HeartflowPlugin(
            types.SimpleNamespace(conversation_manager=ConversationManager()),
            {"enable_heartflow": True},
        )
        calls = 0

        async def summarize(_prompt):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return "summary result"

        plugin._summarize_system_prompt = summarize

        async def scenario():
            return await asyncio.gather(
                plugin._get_or_create_summarized_system_prompt(Event(), "x" * 100),
                plugin._get_or_create_summarized_system_prompt(Event(), "x" * 100),
            )

        results = asyncio.run(scenario())

        self.assertEqual(calls, 1)
        self.assertEqual(results, ["summary result", "summary result"])

    def test_failed_persona_summary_uses_retry_backoff(self):
        class ConversationManager:
            async def get_curr_conversation_id(self, _umo):
                return None

        plugin = plugin_module.HeartflowPlugin(
            types.SimpleNamespace(conversation_manager=ConversationManager()),
            {"enable_heartflow": True},
        )
        calls = 0

        async def summarize(_prompt):
            nonlocal calls
            calls += 1
            return None

        plugin._summarize_system_prompt = summarize
        with patch.object(plugin_module.time, "time", return_value=100.0):
            asyncio.run(plugin._get_or_create_summarized_system_prompt(Event(), "x" * 100))
            asyncio.run(plugin._get_or_create_summarized_system_prompt(Event(), "x" * 100))

        self.assertEqual(calls, 1)
        self.assertEqual(plugin.system_prompt_cache, {})
        self.assertEqual(len(plugin._summary_retry_after), 1)
        self.assertEqual(len(plugin._persona_locks), 0)

    def test_persona_summary_retries_after_backoff_and_new_hash_bypasses_it(self):
        class ConversationManager:
            async def get_curr_conversation_id(self, _umo):
                return None

        plugin = plugin_module.HeartflowPlugin(
            types.SimpleNamespace(conversation_manager=ConversationManager()),
            {"enable_heartflow": True},
        )
        calls = 0

        async def summarize(_prompt):
            nonlocal calls
            calls += 1
            return None

        plugin._summarize_system_prompt = summarize
        with patch.object(plugin_module.time, "time", return_value=100.0):
            asyncio.run(plugin._get_or_create_summarized_system_prompt(Event(), "x" * 100))
            asyncio.run(plugin._get_or_create_summarized_system_prompt(Event(), "y" * 100))
        self.assertEqual(calls, 2)

        with patch.object(
            plugin_module.time,
            "time",
            return_value=100.0 + plugin._SUMMARY_FAILURE_RETRY_SECONDS + 1,
        ):
            asyncio.run(plugin._get_or_create_summarized_system_prompt(Event(), "x" * 100))
        self.assertEqual(calls, 3)

    def test_successful_persona_summary_clears_failure_backoff(self):
        class ConversationManager:
            async def get_curr_conversation_id(self, _umo):
                return None

        plugin = plugin_module.HeartflowPlugin(
            types.SimpleNamespace(conversation_manager=ConversationManager()),
            {"enable_heartflow": True},
        )
        results = iter([None, "summary result"])

        async def summarize(_prompt):
            return next(results)

        prompt = "x" * 100
        key = plugin._persona_cache_key("default", prompt)
        plugin._summarize_system_prompt = summarize
        with patch.object(plugin_module.time, "time", return_value=100.0):
            asyncio.run(plugin._get_or_create_summarized_system_prompt(Event(), prompt))
        with patch.object(
            plugin_module.time,
            "time",
            return_value=100.0 + plugin._SUMMARY_FAILURE_RETRY_SECONDS + 1,
        ):
            result = asyncio.run(plugin._get_or_create_summarized_system_prompt(Event(), prompt))

        self.assertEqual(result, "summary result")
        self.assertNotIn(key, plugin._summary_retry_after)

    def test_persona_summary_request_has_a_source_budget(self):
        class Provider:
            def __init__(self):
                self.prompt = ""

            async def text_chat(self, **kwargs):
                self.prompt = kwargs["prompt"]
                return LLMResponse('{"summarized_persona":"bounded summary"}')

        provider = Provider()
        plugin = plugin_module.HeartflowPlugin(
            types.SimpleNamespace(get_provider_by_id=lambda _name: provider),
            {"enable_heartflow": True, "judge_provider_name": "judge"},
        )

        result = asyncio.run(plugin._summarize_system_prompt("P" * 50000))

        self.assertEqual(result, "bounded summary")
        self.assertLessEqual(len(provider.prompt), 13000)

    def test_activity_uses_recent_timestamps_not_lifetime_total(self):
        plugin = self.make_plugin()
        state = plugin._get_chat_state("group")
        state.total_messages = 1000
        now = plugin_module.time.time()
        old_history = [
            plugin_module.RawMessage("u", str(i), "old", now - 3600, False)
            for i in range(30)
        ]

        context = plugin._build_chat_context_from_snapshot(old_history, state)

        self.assertIn("最近活跃度：低", context)

    def test_idle_chat_state_and_buffer_are_cleaned(self):
        plugin = self.make_plugin()
        plugin.chat_states["old"] = plugin_module.ChatState(last_activity_time=1.0)
        plugin._raw_msg_buffer["old"] = plugin_module.deque(
            [plugin_module.RawMessage("u", "1", "old", 1.0)]
        )

        with patch.object(
            plugin_module.time,
            "time",
            return_value=plugin._CHAT_STATE_TTL_SECONDS + plugin._CLEANUP_INTERVAL_SECONDS + 10,
        ):
            plugin._cleanup_idle_chats()

        self.assertNotIn("old", plugin.chat_states)
        self.assertNotIn("old", plugin._raw_msg_buffer)

    def test_llm_note_is_scoped_and_idempotent(self):
        plugin = self.make_plugin()
        ordinary = Event()
        triggered = Event()
        triggered.set_extra("heartflow_triggered", True)
        ordinary_req = types.SimpleNamespace(system_prompt="persona")
        triggered_req = types.SimpleNamespace(system_prompt="persona")

        asyncio.run(plugin.on_llm_request(ordinary, ordinary_req))
        asyncio.run(plugin.on_llm_request(triggered, triggered_req))
        asyncio.run(plugin.on_llm_request(triggered, triggered_req))

        self.assertEqual(ordinary_req.system_prompt, "persona")
        self.assertEqual(triggered_req.system_prompt.count("本次是你主动参与群聊"), 1)

    def test_schema_matches_new_safe_defaults(self):
        schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8"))

        self.assertEqual(schema["context_messages_count"]["default"], 10)
        self.assertEqual(schema["max_message_chars"]["default"], 1200)
        self.assertEqual(schema["max_context_chars"]["default"], 6000)
        self.assertFalse(schema["judge_include_reasoning"]["default"])
        self.assertEqual(schema["judge_max_retries"]["default"], 1)
        self.assertNotIn("judge_context_count", schema)


if __name__ == "__main__":
    unittest.main()
