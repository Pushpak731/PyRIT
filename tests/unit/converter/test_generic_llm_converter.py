# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_mock_target_identifier

from pyrit.converter import (
    LLMGenericTextConverter,
    MaliciousQuestionGeneratorConverter,
    NoiseConverter,
    ScientificTranslationConverter,
    TenseConverter,
    ToneConverter,
    TranslationConverter,
    VariationConverter,
)
from pyrit.exceptions import InvalidJsonException
from pyrit.models import Message, MessagePiece, SeedPrompt
from pyrit.prompt_target.common.prompt_target import PromptTarget


@pytest.fixture
def mock_target() -> MagicMock:
    target = MagicMock(spec=PromptTarget)
    response = Message(
        message_pieces=[
            MessagePiece(
                role="assistant",
                original_value="prompt value",
            )
        ]
    )
    target.send_prompt_async = AsyncMock(return_value=[response])
    target.get_identifier.return_value = get_mock_target_identifier("MockLLMTarget")
    return target


async def test_noise_converter_sets_system_prompt_default(mock_target) -> None:
    converter = NoiseConverter(converter_target=mock_target)
    await converter.convert_async(prompt="being awesome")

    mock_target.set_system_prompt_async.assert_called_once()

    system_arg = mock_target.set_system_prompt_async.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert "Grammar error, Delete random letter" in system_arg


async def test_noise_converter_sets_system_prompt(mock_target) -> None:
    converter = NoiseConverter(converter_target=mock_target, noise="extra random periods")
    await converter.convert_async(prompt="being awesome")

    mock_target.set_system_prompt_async.assert_called_once()

    system_arg = mock_target.set_system_prompt_async.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert "extra random periods" in system_arg


async def test_tone_converter_sets_system_prompt(mock_target) -> None:
    converter = ToneConverter(tone="formal", converter_target=mock_target)
    await converter.convert_async(prompt="being awesome")

    mock_target.set_system_prompt_async.assert_called_once()

    system_arg = mock_target.set_system_prompt_async.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert "formal" in system_arg


async def test_tense_converter_sets_system_prompt(mock_target) -> None:
    converter = TenseConverter(tense="past", converter_target=mock_target)
    await converter.convert_async(prompt="being awesome")

    mock_target.set_system_prompt_async.assert_called_once()

    system_arg = mock_target.set_system_prompt_async.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert "past" in system_arg


async def test_malicious_question_converter_sets_system_prompt(mock_target) -> None:
    converter = MaliciousQuestionGeneratorConverter(converter_target=mock_target)
    await converter.convert_async(prompt="being awesome")

    mock_target.set_system_prompt_async.assert_called_once()

    system_arg = mock_target.set_system_prompt_async.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert "Please act as an expert in this domain: being awesome" in system_arg


def test_generic_llm_converter_input_supported() -> None:
    target = MagicMock(spec=PromptTarget)
    converter = LLMGenericTextConverter(converter_target=target)
    assert converter.input_supported("text") is True
    assert converter.input_supported("audio_path") is False


def test_generic_llm_converter_user_prompt_without_objective_raises() -> None:
    target = MagicMock(spec=PromptTarget)
    user_template = MagicMock()
    with pytest.raises(ValueError):
        LLMGenericTextConverter(converter_target=target, user_prompt_template_with_objective=user_template)


def test_generic_llm_converter_init_default_templates_empty() -> None:
    target = MagicMock(spec=PromptTarget)
    converter = LLMGenericTextConverter(converter_target=target)
    assert converter._system_prompt_template is None
    assert converter._user_prompt_template_with_objective is None


def test_generic_llm_converter_default_no_retry_exceptions() -> None:
    target = MagicMock(spec=PromptTarget)
    converter = LLMGenericTextConverter(converter_target=target)
    assert converter._retry_exceptions == ()


def test_generic_llm_converter_class_attr_retry_exceptions() -> None:
    class _RetryingConverter(LLMGenericTextConverter):
        RETRY_EXCEPTIONS = (ValueError,)

    target = MagicMock(spec=PromptTarget)
    converter = _RetryingConverter(converter_target=target)
    assert converter._retry_exceptions == (ValueError,)


def test_generic_llm_converter_instance_retry_exceptions_overrides_class_attr() -> None:
    class _RetryingConverter(LLMGenericTextConverter):
        RETRY_EXCEPTIONS = (ValueError,)

    target = MagicMock(spec=PromptTarget)
    converter = _RetryingConverter(converter_target=target, retry_exceptions=(KeyError,))
    assert converter._retry_exceptions == (KeyError,)


async def test_convert_async_no_user_template_sets_only_original_value(mock_target) -> None:
    converter = LLMGenericTextConverter(converter_target=mock_target)
    await converter.convert_async(prompt="hello")

    sent_message = mock_target.send_prompt_async.call_args[1]["message"]
    piece = sent_message.message_pieces[0]
    assert piece.original_value == "hello"
    assert piece.converted_value == "hello"


async def test_convert_async_with_user_template_preserves_original_and_renders_converted(mock_target) -> None:
    user_template = SeedPrompt(
        value="Wrap: [{{ objective }}]",
        parameters=["objective"],
        data_type="text",
    )
    converter = LLMGenericTextConverter(converter_target=mock_target, user_prompt_template_with_objective=user_template)
    await converter.convert_async(prompt="raw input")

    sent_message = mock_target.send_prompt_async.call_args[1]["message"]
    piece = sent_message.message_pieces[0]
    assert piece.original_value == "raw input"
    assert piece.converted_value == "Wrap: [raw input]"


async def test_convert_async_user_template_receives_extra_kwargs(mock_target) -> None:
    user_template = SeedPrompt(
        value="Lang={{ language }} Obj={{ objective }}",
        parameters=["objective", "language"],
        data_type="text",
    )
    converter = LLMGenericTextConverter(
        converter_target=mock_target,
        user_prompt_template_with_objective=user_template,
        language="spanish",
    )
    await converter.convert_async(prompt="hello")

    sent_message = mock_target.send_prompt_async.call_args[1]["message"]
    assert sent_message.message_pieces[0].converted_value == "Lang=spanish Obj=hello"


async def test_convert_async_input_validation_raises_before_set_system_prompt(mock_target) -> None:
    system_template = SeedPrompt(value="sys", data_type="text")
    converter = LLMGenericTextConverter(converter_target=mock_target, system_prompt_template=system_template)
    with pytest.raises(ValueError, match="Input type not supported"):
        await converter.convert_async(prompt="hello", input_type="image_path")
    mock_target.set_system_prompt_async.assert_not_called()
    mock_target.send_prompt_async.assert_not_called()


async def test_convert_async_process_response_hook_called(mock_target) -> None:
    class _UpperCaseConverter(LLMGenericTextConverter):
        def _process_response(self, response_text: str) -> str:
            return response_text.upper()

    converter = _UpperCaseConverter(converter_target=mock_target)
    result = await converter.convert_async(prompt="anything")
    assert result.output_text == "PROMPT VALUE"


async def test_send_with_retries_no_retry_when_empty_exception_tuple(mock_target) -> None:
    converter = LLMGenericTextConverter(converter_target=mock_target)
    mock_target.send_prompt_async.side_effect = ValueError("boom")
    with pytest.raises(ValueError, match="boom"):
        await converter.convert_async(prompt="hello")
    assert mock_target.send_prompt_async.call_count == 1


async def test_send_with_retries_retries_on_configured_exception(mock_target) -> None:
    class _RetryingConverter(LLMGenericTextConverter):
        RETRY_EXCEPTIONS = (InvalidJsonException,)

        def _process_response(self, response_text: str) -> str:
            raise InvalidJsonException(message="bad")

    converter = _RetryingConverter(converter_target=mock_target)
    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(InvalidJsonException):
            await converter.convert_async(prompt="hello")
    assert mock_target.send_prompt_async.call_count == 2  # RETRY_MAX_NUM_ATTEMPTS=2 in conftest


async def test_send_with_retries_does_not_retry_unrelated_exception(mock_target) -> None:
    class _RetryingConverter(LLMGenericTextConverter):
        RETRY_EXCEPTIONS = (InvalidJsonException,)

    converter = _RetryingConverter(converter_target=mock_target)
    mock_target.send_prompt_async.side_effect = ValueError("not retried")
    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(ValueError, match="not retried"):
            await converter.convert_async(prompt="hello")
    assert mock_target.send_prompt_async.call_count == 1


async def test_send_with_retries_succeeds_after_one_failure(mock_target) -> None:
    class _RetryingConverter(LLMGenericTextConverter):
        RETRY_EXCEPTIONS = (InvalidJsonException,)
        _calls = 0

        def _process_response(self, response_text: str) -> str:
            type(self)._calls += 1
            if type(self)._calls == 1:
                raise InvalidJsonException(message="first")
            return response_text

    converter = _RetryingConverter(converter_target=mock_target)
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await converter.convert_async(prompt="hello")
    assert result.output_text == "prompt value"
    assert mock_target.send_prompt_async.call_count == 2


async def test_send_with_retries_uses_static_attempt_count_when_provided(mock_target) -> None:
    class _RetryingConverter(LLMGenericTextConverter):
        RETRY_EXCEPTIONS = (InvalidJsonException,)

        def _process_response(self, response_text: str) -> str:
            raise InvalidJsonException(message="bad")

    converter = _RetryingConverter(converter_target=mock_target, max_retry_attempts=4)
    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(InvalidJsonException):
            await converter.convert_async(prompt="hello")
    assert mock_target.send_prompt_async.call_count == 4


async def test_send_with_retries_no_wait_by_default(mock_target) -> None:
    """Default wait is none (0 seconds), matching pyrit_json_retry behavior."""

    class _RetryingConverter(LLMGenericTextConverter):
        RETRY_EXCEPTIONS = (InvalidJsonException,)

        def _process_response(self, response_text: str) -> str:
            raise InvalidJsonException(message="bad")

    converter = _RetryingConverter(converter_target=mock_target)
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(InvalidJsonException):
            await converter.convert_async(prompt="hello")
    for call in mock_sleep.call_args_list:
        assert call.args[0] == 0.0


async def test_send_with_retries_uses_exponential_wait_when_max_seconds_provided(mock_target) -> None:
    class _RetryingConverter(LLMGenericTextConverter):
        RETRY_EXCEPTIONS = (InvalidJsonException,)

        def _process_response(self, response_text: str) -> str:
            raise InvalidJsonException(message="bad")

    converter = _RetryingConverter(converter_target=mock_target, retry_wait_max_seconds=10)
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(InvalidJsonException):
            await converter.convert_async(prompt="hello")
    # waits should be > 0 between attempts (exponential backoff)
    nonzero_waits = [c for c in mock_sleep.call_args_list if c.args[0] > 0]
    assert len(nonzero_waits) >= 1


def _custom_template(value: str, *parameters: str) -> SeedPrompt:
    return SeedPrompt(value=value, data_type="text", parameters=list(parameters))


@pytest.mark.parametrize(
    "converter_class, kwargs, template",
    [
        (ToneConverter, {"tone": "angry"}, _custom_template("Rewrite in a {{ tone }} tone, in French.", "tone")),
        (TenseConverter, {"tense": "past"}, _custom_template("Use the {{ tense }} tense, as a poem.", "tense")),
        (
            TranslationConverter,
            {"language": "french"},
            _custom_template("Translate to {{ language }} slang.", "language"),
        ),
        (
            NoiseConverter,
            {},
            _custom_template("Add {{ number_errors }} typos: {{ noise }}", "noise", "number_errors"),
        ),
        (ScientificTranslationConverter, {"mode": "academic"}, _custom_template("Rewrite as a lab report.")),
        (
            VariationConverter,
            {},
            _custom_template("Give {{ number_iterations }} rhyming variations.", "number_iterations"),
        ),
    ],
)
def test_llm_converter_identifier_includes_custom_prompt_template(
    mock_target, converter_class, kwargs, template
) -> None:
    default = converter_class(converter_target=mock_target, **kwargs)
    custom = converter_class(converter_target=mock_target, prompt_template=template, **kwargs)
    same_custom = converter_class(converter_target=mock_target, prompt_template=template, **kwargs)

    default_params = default.get_identifier().params
    custom_params = custom.get_identifier().params
    assert default_params["system_prompt_template_hash"] != custom_params["system_prompt_template_hash"]
    assert default.get_identifier().hash != custom.get_identifier().hash
    assert custom.get_identifier().hash == same_custom.get_identifier().hash
    for key, value in kwargs.items():
        assert custom_params[key] == value


def _kwargs_template() -> SeedPrompt:
    return SeedPrompt(
        value="Translate the text into {{ language }}.",
        data_type="text",
        parameters=["language"],
    )


def test_generic_llm_converter_identifier_distinguishes_template_kwargs(mock_target) -> None:
    """Converters sharing a template but passing different template parameters
    render different prompts, so they must not share an identifier (#3385)."""
    template = _kwargs_template()
    french = LLMGenericTextConverter(converter_target=mock_target, system_prompt_template=template, language="French")
    spanish = LLMGenericTextConverter(converter_target=mock_target, system_prompt_template=template, language="Spanish")

    assert french.get_identifier().unique_name != spanish.get_identifier().unique_name
    assert french.get_identifier().hash != spanish.get_identifier().hash


def test_generic_llm_converter_identifier_same_kwargs_is_stable(mock_target) -> None:
    template = _kwargs_template()
    french = LLMGenericTextConverter(converter_target=mock_target, system_prompt_template=template, language="French")
    french_again = LLMGenericTextConverter(
        converter_target=mock_target, system_prompt_template=template, language="French"
    )

    assert french.get_identifier().hash == french_again.get_identifier().hash


def test_generic_llm_converter_identifier_exposes_template_kwargs(mock_target) -> None:
    template = _kwargs_template()
    converter = LLMGenericTextConverter(
        converter_target=mock_target, system_prompt_template=template, language="French"
    )

    assert converter.get_identifier().params["prompt_template_kwargs"] == {"language": "French"}


def test_generic_llm_converter_identifier_without_kwargs_is_none(mock_target) -> None:
    converter = LLMGenericTextConverter(converter_target=mock_target)

    assert converter.get_identifier().params.get("prompt_template_kwargs") is None


def test_generic_llm_converter_identifier_distinguishes_user_template_kwargs(mock_target) -> None:
    """The parameters reach the user prompt template too, so converters built
    around a user template must also be told apart by them."""
    template = SeedPrompt(
        value="Rewrite {{ objective }} in {{ style }}.",
        data_type="text",
        parameters=["objective", "style"],
    )
    formal = LLMGenericTextConverter(
        converter_target=mock_target, user_prompt_template_with_objective=template, style="formal"
    )
    casual = LLMGenericTextConverter(
        converter_target=mock_target, user_prompt_template_with_objective=template, style="casual"
    )

    assert formal.get_identifier().hash != casual.get_identifier().hash


def test_generic_llm_converter_identifier_distinguishes_structured_kwargs(mock_target) -> None:
    """Template parameters can be lists or dicts; nested values still tell
    converters apart and stay JSON-serializable for storage."""
    template = SeedPrompt(
        value="Follow {{ rules }}.",
        data_type="text",
        parameters=["rules"],
    )
    short = LLMGenericTextConverter(
        converter_target=mock_target,
        system_prompt_template=template,
        rules=["be brief", "be kind"],
    )
    long = LLMGenericTextConverter(
        converter_target=mock_target,
        system_prompt_template=template,
        rules=["be brief", "be kind", "be bold"],
    )

    assert short.get_identifier().hash != long.get_identifier().hash
    assert short.get_identifier().params["prompt_template_kwargs"] == {"rules": ["be brief", "be kind"]}


def test_registry_accepts_converters_differing_only_by_kwargs(mock_target, patch_central_database) -> None:
    """The issue's failure mode: registering a second converter that differs
    only by its template parameters raised a duplicate-name error."""
    from pyrit.registry import ConverterRegistry

    template = _kwargs_template()
    french = LLMGenericTextConverter(converter_target=mock_target, system_prompt_template=template, language="French")
    spanish = LLMGenericTextConverter(converter_target=mock_target, system_prompt_template=template, language="Spanish")

    registry = ConverterRegistry()
    registry.instances.register(french)
    registry.instances.register(spanish)


def test_subclass_identifier_still_promotes_its_own_parameter(mock_target) -> None:
    """ToneConverter and friends keep their readable promoted parameter next to
    the new nested kwargs view."""
    converter = ToneConverter(converter_target=mock_target, tone="excited")
    tone_again = ToneConverter(converter_target=mock_target, tone="excited")
    other_tone = ToneConverter(converter_target=mock_target, tone="serious")

    assert converter.get_identifier().params["tone"] == "excited"
    assert converter.get_identifier().hash == tone_again.get_identifier().hash
    assert converter.get_identifier().hash != other_tone.get_identifier().hash
