from types import SimpleNamespace

from saas_bench.agents.bash_agent.agent import BashAgent, Message


class _Block:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Response:
    def __init__(self, content, stop_reason="tool_use"):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
        )


class _Stream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_final_message(self):
        return self.response


class _Messages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _Stream(self.responses.pop(0))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class Anthropic:
    def __init__(self, responses=()):
        self.messages = _Messages(responses)


def _agent(model="claude-fable-5", reasoning_effort="high", client=None):
    return BashAgent(
        tool_descriptions=[],
        client=client or Anthropic(),
        model=model,
        system_prompt="system",
        reasoning_effort=reasoning_effort,
    )


def test_fable_uses_adaptive_effort_without_output_128k_beta_header():
    agent = _agent(model="claude-fable-5", reasoning_effort="high")
    api_kwargs = {}

    agent._apply_anthropic_reasoning_params(api_kwargs)

    assert api_kwargs["thinking"] == {"type": "adaptive"}
    assert api_kwargs["output_config"] == {"effort": "high"}
    assert agent._anthropic_extra_headers() == {}


def test_bedrock_fable_id_is_detected_as_fable():
    agent = _agent(model="anthropic.claude-fable-5")

    assert agent._is_anthropic_fable_or_mythos_model()
    assert agent._anthropic_extra_headers() == {}


def test_non_fable_anthropic_models_keep_output_128k_beta_header():
    agent = _agent(model="claude-sonnet-4-6")

    assert agent._anthropic_extra_headers() == {
        "anthropic-beta": "output-128k-2025-02-19",
    }


def test_haiku_uses_fixed_extended_thinking_budget():
    agent = _agent(model="claude-haiku-4-5", reasoning_effort="medium")
    api_kwargs = {}

    agent._apply_anthropic_reasoning_params(api_kwargs)

    assert api_kwargs["thinking"] == {
        "type": "enabled",
        "budget_tokens": 16000,
    }
    assert "output_config" not in api_kwargs


def test_fable_refusal_retries_and_returns_tool_action():
    client = Anthropic([
        _Response([
            _Block("text", text="I cannot comply with that request."),
        ], stop_reason="refusal"),
        _Response([
            _Block(
                "tool_use",
                id="toolu_1",
                name="bash",
                input={"command": "pwd"},
            ),
        ]),
    ])
    agent = _agent(model="claude-fable-5", reasoning_effort="low", client=client)
    agent.conversation.append(Message(role="user", content="dashboard"))

    action = agent._call_anthropic()

    assert action.tool == "bash"
    assert action.arguments == {"command": "pwd"}
    assert len(client.messages.calls) == 2
    assert "extra_headers" not in client.messages.calls[0]
    assert any(
        msg.role == "user" and "refusal" in str(msg.content).lower()
        for msg in agent.conversation
    )
