"""
Built-in chat template registry.

Provides ``TemplateInfo`` (name, Jinja2 string, special tokens) for ~10
popular model families and a ``TemplateRegistry`` to look them up.

Usage::

    from chat_template_tester.templates import registry

    # List all available templates
    for info in registry.list():
        print(info.name, info.description)

    # Get one by name
    tmpl = registry.get("llama3")

    # Custom template
    from chat_template_tester.templates import TemplateInfo
    registry.register(TemplateInfo(
        name="my-custom",
        template="{{ messages | tojson }}",
        bos_token="<s>",
        eos_token="</s>",
        description="My custom template",
    ))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── TemplateInfo ──────────────────────────────────────────────────


@dataclass
class TemplateInfo:
    """A chat template definition.

    Attributes:
        name: Short identifier (e.g. ``"llama3"``).
        template: Jinja2 template string (as stored in
            ``tokenizer_config.json`` ``chat_template`` field).
        bos_token: Beginning-of-sequence token (or ``None``).
        eos_token: End-of-sequence token (or ``None``).
        description: Human-readable description.
    """

    name: str
    template: str
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    description: str = ""
    stop_tokens: List[str] = field(default_factory=list)


# ── Template definitions ──────────────────────────────────────────

# Each template is the actual Jinja2 string from the model's
# tokenizer_config.json on Hugging Face, cleaned up for readability.

TEMPLATE_LLAMA3 = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
    "+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = bos_token + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)

TEMPLATE_LLAMA2 = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] | trim + '\\n\\n' %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% set system_message = '' %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate') }}"
    "{% endif %}"
    "{% if loop.index0 == 0 and system_message %}"
    "{% set content = bos_token + system_message + '[INST] ' + message['content'] | trim + ' [/INST]' %}"
    "{% else %}"
    "{% set content = bos_token + '[INST] ' + message['content'] | trim + ' [/INST]' %}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ content }}"
    "{% else %}"
    "{% set content = ' ' + message['content'] | trim + ' ' + eos_token %}"
    "{{ content }}"
    "{% endif %}"
    "{% endfor %}"
)

TEMPLATE_CHATML = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)

TEMPLATE_MISTRAL = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'] | trim + ' [/INST]' }}"
    "{% else %}"
    "{{ message['content'] | trim + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)

TEMPLATE_VICUNA = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] | trim + '\\n\\n' %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% set system_message = '' %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate') }}"
    "{% endif %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = system_message + 'USER: ' + message['content'] | trim %}"
    "{% else %}"
    "{% set content = '\\nUSER: ' + message['content'] | trim %}"
    "{% endif %}"
    "{{ content }}"
    "{% if (message['role'] == 'assistant') %}"
    "{% set content = '\\nASSISTANT: ' + message['content'] | trim + '</s>' %}"
    "{{ content }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\\nASSISTANT: ' }}"
    "{% endif %}"
)

TEMPLATE_GEMMA = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
    "{% else %}"
    "{{ '<start_of_turn>model\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>model\\n' }}"
    "{% endif %}"
)

TEMPLATE_PHI3 = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{'<|user|>\\n' + message['content'] + '<|end|>\\n'}}"
    "{% elif message['role'] == 'assistant' %}"
    "{{'<|assistant|>\\n' + message['content'] + '<|end|>\\n'}}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|assistant|>\\n' }}"
    "{% endif %}"
)

TEMPLATE_DEEPSEEK = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ 'User: ' + message['content'] + '\\n\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'Assistant: ' + message['content'] + '<｜end▁of▁sentence｜>' }}"
    "{% elif message['role'] == 'system' %}"
    "{{ message['content'] + '\\n\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'Assistant: ' }}"
    "{% endif %}"
)

TEMPLATE_COMMAND_R = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<BOS_TOKEN>' }}"
    "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + message['content'] | trim + '<|END_OF_TURN_TOKEN|>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + message['content'] | trim + '<|END_OF_TURN_TOKEN|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' + message['content'] | trim + '<|END_OF_TURN_TOKEN|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}"
    "{% endif %}"
)

TEMPLATE_QWEN25 = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)


# ── Registry ──────────────────────────────────────────────────────


class TemplateRegistry:
    """Registry of named chat templates."""

    def __init__(self) -> None:
        self._templates: Dict[str, TemplateInfo] = {}

    def register(self, info: TemplateInfo) -> None:
        """Register a template (replaces any existing with the same name)."""
        self._templates[info.name] = info

    def get(self, name: str) -> TemplateInfo:
        """Look up a template by name.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._templates:
            raise KeyError(f"Unknown template: {name!r}. "
                           f"Available: {', '.join(self._templates)}")
        return self._templates[name]

    def list(self) -> List[TemplateInfo]:
        """Return all registered templates (sorted by name)."""
        return sorted(self._templates.values(), key=lambda t: t.name)

    def __contains__(self, name: str) -> bool:
        return name in self._templates


# ── Built-in templates ────────────────────────────────────────────

_BUILTINS: List[TemplateInfo] = [
    TemplateInfo(
        name="llama3",
        template=TEMPLATE_LLAMA3,
        bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>",
        stop_tokens=["<|eot_id|>", "<|start_header_id|>"],
        description="Meta Llama 3 / 3.1 — header-based roles with <|eot_id|> separator.",
    ),
    TemplateInfo(
        name="llama2",
        template=TEMPLATE_LLAMA2,
        bos_token="<s>",
        eos_token="</s>",
        stop_tokens=["</s>"],
        description="Meta Llama 2 — [INST] / [/INST] style with optional system message.",
    ),
    TemplateInfo(
        name="chatml",
        template=TEMPLATE_CHATML,
        bos_token=None,
        eos_token="<|im_end|>",
        stop_tokens=["<|im_end|>"],
        description="OpenAI ChatML — <|im_start|>role / <|im_end|> markers. Used by Qwen, many fine-tunes.",
    ),
    TemplateInfo(
        name="mistral",
        template=TEMPLATE_MISTRAL,
        bos_token="<s>",
        eos_token="</s>",
        stop_tokens=["</s>"],
        description="Mistral AI — [INST] style without system prompt support.",
    ),
    TemplateInfo(
        name="vicuna",
        template=TEMPLATE_VICUNA,
        bos_token=None,
        eos_token="</s>",
        stop_tokens=["</s>"],
        description="Vicuna 1.1 — USER: / ASSISTANT: style with optional system prefix.",
    ),
    TemplateInfo(
        name="gemma",
        template=TEMPLATE_GEMMA,
        bos_token="<bos>",
        eos_token="<eos>",
        stop_tokens=["<eos>", "<end_of_turn>"],
        description="Google Gemma — <start_of_turn>user|model / <end_of_turn> markers.",
    ),
    TemplateInfo(
        name="phi3",
        template=TEMPLATE_PHI3,
        bos_token=None,
        eos_token="<|end|>",
        stop_tokens=["<|end|>"],
        description="Microsoft Phi-3 — <|user|> / <|assistant|> / <|end|> markers.",
    ),
    TemplateInfo(
        name="deepseek",
        template=TEMPLATE_DEEPSEEK,
        bos_token="<｜begin▁of▁sentence｜>",
        eos_token="<｜end▁of▁sentence｜>",
        stop_tokens=["<｜end▁of▁sentence｜>"],
        description="DeepSeek V2/V3 — 'User:' / 'Assistant:' labels with unicode special tokens.",
    ),
    TemplateInfo(
        name="command-r",
        template=TEMPLATE_COMMAND_R,
        bos_token="<BOS_TOKEN>",
        eos_token=None,
        stop_tokens=["<|END_OF_TURN_TOKEN|>"],
        description="Cohere Command R — <|START_OF_TURN_TOKEN|> with SYSTEM/USER/CHATBOT role tokens.",
    ),
    TemplateInfo(
        name="qwen2.5",
        template=TEMPLATE_QWEN25,
        bos_token=None,
        eos_token="<|im_end|>",
        stop_tokens=["<|im_end|>"],
        description="Qwen 2.5 — ChatML variant with automatic system prompt injection.",
    ),
]

# Populate the default registry
registry = TemplateRegistry()
for _info in _BUILTINS:
    registry.register(_info)
