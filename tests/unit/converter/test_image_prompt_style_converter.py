# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from unit.mocks import MockPromptTarget, get_mock_target_identifier

from pyrit.common.random_context import configure_random_seed
from pyrit.converter import ImagePromptStyleConverter
from pyrit.memory import SQLiteMemory
from pyrit.models import ConverterIdentifier, Message, MessagePiece
from pyrit.prompt_normalizer import ConverterConfiguration, PromptNormalizer
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.registry import ConverterRegistry


@pytest.fixture
def mock_target() -> MagicMock:
    target = MagicMock(spec=PromptTarget)
    response = Message(
        message_pieces=[
            MessagePiece(
                role="assistant",
                original_value="A blurry bodycam shot of a figure in a dark alley",
            )
        ]
    )
    target.send_prompt_async = AsyncMock(return_value=[response])
    target.get_identifier.return_value = get_mock_target_identifier("MockLLMTarget")
    return target


def test_init_valid_filter_and_variation(mock_target) -> None:
    converter = ImagePromptStyleConverter(
        converter_target=mock_target,
        filter_name="gritty_documentary",
        variation="bodycam_footage",
    )
    assert converter._filter_name == "gritty_documentary"
    assert converter._variation == "bodycam_footage"
    assert "bodycam_footage" in converter._variation_map


def test_init_no_filter_picks_random(mock_target) -> None:
    converter = ImagePromptStyleConverter(
        converter_target=mock_target,
    )
    available = ImagePromptStyleConverter.list_available_filters()
    assert converter._filter_name in available
    assert converter._variation is None


def test_init_no_filter_uses_configured_root_seed(mock_target) -> None:
    try:
        configure_random_seed(seed=42)
        first = ImagePromptStyleConverter(converter_target=mock_target)
        second = ImagePromptStyleConverter(converter_target=mock_target)

        assert first._filter_name == second._filter_name
    finally:
        configure_random_seed(seed=None)


def test_init_filter_path_custom_yaml(mock_target, tmp_path) -> None:
    custom_yaml = tmp_path / "custom_filter.yaml"
    custom_yaml.write_text("style_instructions: custom style\nvariations:\n  my_variation: description of variation\n")
    converter = ImagePromptStyleConverter(
        converter_target=mock_target,
        filter_path=custom_yaml,
        variation="my_variation",
    )
    assert converter._filter_name == "custom_filter"
    assert "my_variation" in converter._variation_map


def test_init_filter_path_nonexistent_raises(mock_target) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        ImagePromptStyleConverter(
            converter_target=mock_target,
            filter_path="/nonexistent/path.yaml",
        )


def test_init_both_filter_name_and_path_raises(mock_target, tmp_path) -> None:
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text("style_instructions: style\nvariations:\n  V1: desc\n")
    with pytest.raises(ValueError, match="Only one of"):
        ImagePromptStyleConverter(
            converter_target=mock_target,
            filter_name="gritty_documentary",
            filter_path=custom_yaml,
        )


def test_init_variation_none_is_valid(mock_target) -> None:
    converter = ImagePromptStyleConverter(
        converter_target=mock_target,
        filter_name="gritty_documentary",
    )
    assert converter._variation is None


def test_init_variation_not_case_sensitive(mock_target) -> None:
    converter = ImagePromptStyleConverter(
        converter_target=mock_target,
        filter_name="gritty_documentary",
        variation="BODYCAM_FOOTAGE",
    )
    assert converter._variation == "BODYCAM_FOOTAGE"
    assert "bodycam_footage" in converter._variation_map


def test_init_invalid_filter_name_raises(mock_target) -> None:
    with pytest.raises(ValueError, match="not found"):
        ImagePromptStyleConverter(
            converter_target=mock_target,
            filter_name="nonexistent_filter",
        )


def test_init_invalid_variation_raises(mock_target) -> None:
    with pytest.raises(ValueError, match="not found in filter"):
        ImagePromptStyleConverter(
            converter_target=mock_target,
            filter_name="gritty_documentary",
            variation="Nonexistent Variation",
        )


def test_init_filter_missing_style_instructions_raises(mock_target, tmp_path) -> None:
    bad_yaml = tmp_path / "missing_style.yaml"
    bad_yaml.write_text("variations:\n  v1: desc\n")
    with pytest.raises(ValueError, match="missing required key 'style_instructions'"):
        ImagePromptStyleConverter(converter_target=mock_target, filter_path=bad_yaml)


def test_init_filter_missing_variations_raises(mock_target, tmp_path) -> None:
    bad_yaml = tmp_path / "missing_variations.yaml"
    bad_yaml.write_text("style_instructions: some style\n")
    with pytest.raises(ValueError, match="missing required key 'variations'"):
        ImagePromptStyleConverter(converter_target=mock_target, filter_path=bad_yaml)


def test_init_filter_empty_variations_raises(mock_target, tmp_path) -> None:
    bad_yaml = tmp_path / "empty_variations.yaml"
    bad_yaml.write_text("style_instructions: some style\nvariations: {}\n")
    with pytest.raises(ValueError, match="non-empty mapping"):
        ImagePromptStyleConverter(converter_target=mock_target, filter_path=bad_yaml)


def test_init_filter_non_mapping_top_level_raises(mock_target, tmp_path) -> None:
    bad_yaml = tmp_path / "list_top_level.yaml"
    bad_yaml.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="expected a YAML mapping at the top level"):
        ImagePromptStyleConverter(converter_target=mock_target, filter_path=bad_yaml)


def test_init_filter_style_instructions_wrong_type_raises(mock_target, tmp_path) -> None:
    bad_yaml = tmp_path / "bad_style_type.yaml"
    bad_yaml.write_text("style_instructions:\n  - not\n  - a\n  - string\nvariations:\n  v1: desc\n")
    with pytest.raises(ValueError, match="must be a string"):
        ImagePromptStyleConverter(converter_target=mock_target, filter_path=bad_yaml)


def test_list_available_filters() -> None:
    filters = ImagePromptStyleConverter.list_available_filters()
    assert isinstance(filters, list)
    assert "gritty_documentary" in filters
    assert len(filters) > 0


async def test_convert_async_with_specific_variation(mock_target) -> None:
    converter = ImagePromptStyleConverter(
        converter_target=mock_target,
        filter_name="gritty_documentary",
        variation="bodycam_footage",
    )
    result = await converter.convert_async(prompt="person walking through a dark alley")

    mock_target.set_system_prompt_async.assert_called_once()
    system_arg = mock_target.set_system_prompt_async.call_args[1]["system_prompt"]
    assert "bodycam_footage" in system_arg
    assert "style_instructions" not in system_arg or "CRITICAL INSTRUCTION" in system_arg

    mock_target.send_prompt_async.assert_called_once()
    assert result.output_text == "A blurry bodycam shot of a figure in a dark alley"
    assert result.output_type == "text"


async def test_convert_async_with_random_variation(mock_target) -> None:
    converter = ImagePromptStyleConverter(
        converter_target=mock_target,
        filter_name="gritty_documentary",
    )
    result = await converter.convert_async(prompt="person in a park")

    mock_target.set_system_prompt_async.assert_called_once()
    system_arg = mock_target.set_system_prompt_async.call_args[1]["system_prompt"]
    # Should contain one of the variation names
    assert any(name in system_arg for name in converter._variations)

    assert result.output_text == "A blurry bodycam shot of a figure in a dark alley"


async def test_convert_async_unsupported_input_type_raises(mock_target) -> None:
    converter = ImagePromptStyleConverter(
        converter_target=mock_target,
        filter_name="gritty_documentary",
    )
    with pytest.raises(ValueError, match="Input type not supported"):
        await converter.convert_async(prompt="/tmp/image.png", input_type="image_path")


def test_duplicate_variation_prefix_logs_warning(mock_target, caplog) -> None:
    """Duplicate variation keys (case-insensitive) should log a warning but not raise."""
    from unittest.mock import MagicMock, mock_open, patch

    duplicate_yaml = {
        "style_instructions": "test style",
        "variations": {
            "bodycam_footage": "first version",
            "BODYCAM_FOOTAGE": "second version",
        },
    }

    mock_seed_prompt = MagicMock()

    with (
        caplog.at_level("WARNING", logger="pyrit.converter.image_prompt_style_converter"),
        patch(
            "pyrit.converter.image_prompt_style_converter.SeedPrompt.from_yaml_file",
            return_value=mock_seed_prompt,
        ),
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open()),
        patch("pyrit.converter.image_prompt_style_converter.safe_load_yaml", return_value=duplicate_yaml),
    ):
        converter = ImagePromptStyleConverter(
            converter_target=mock_target,
            filter_name="gritty_documentary",
        )

    assert "Duplicate variation key" in caplog.text
    assert converter._variation_map["bodycam_footage"] == "BODYCAM_FOOTAGE"


def _write_filter(path, instructions: str, variation: str = "close_up: tight framing") -> None:
    name, description = variation.split(": ", 1)
    path.write_text(
        f"style_instructions: {instructions}\nvariations:\n  {name}: {description}\n",
        encoding="utf-8",
    )


def test_identifier_distinguishes_same_named_custom_filters(mock_target, tmp_path) -> None:
    """Two custom filter files with the same stem but different content render
    different prompts, so they must not share an identifier (#3386)."""
    charcoal = tmp_path / "team_a" / "gritty_documentary.yaml"
    charcoal.parent.mkdir()
    _write_filter(charcoal, "Render as a charcoal sketch.")
    neon = tmp_path / "team_b" / "gritty_documentary.yaml"
    neon.parent.mkdir()
    _write_filter(neon, "Render as neon cyberpunk.")

    charcoal_converter = ImagePromptStyleConverter(converter_target=mock_target, filter_path=charcoal)
    neon_converter = ImagePromptStyleConverter(converter_target=mock_target, filter_path=neon)

    assert charcoal_converter.get_identifier().hash != neon_converter.get_identifier().hash
    assert charcoal_converter.get_identifier().unique_name != neon_converter.get_identifier().unique_name


def test_identifier_distinguishes_custom_filter_from_same_named_builtin(mock_target, tmp_path) -> None:
    """A custom file named after a built-in filter must not share the built-in's
    identifier unless its content is identical (#3386)."""
    custom = tmp_path / "gritty_documentary.yaml"
    _write_filter(custom, "Render as a charcoal sketch.")

    custom_converter = ImagePromptStyleConverter(converter_target=mock_target, filter_path=custom)
    builtin_converter = ImagePromptStyleConverter(converter_target=mock_target, filter_name="gritty_documentary")

    assert custom_converter.get_identifier().hash != builtin_converter.get_identifier().hash


def test_identifier_identical_content_is_stable(mock_target, tmp_path) -> None:
    """Two files with the same stem and identical content behave identically,
    so their identifiers match."""
    first = tmp_path / "a" / "noir.yaml"
    first.parent.mkdir()
    _write_filter(first, "Shoot in black and white.")
    second = tmp_path / "b" / "noir.yaml"
    second.parent.mkdir()
    _write_filter(second, "Shoot in black and white.")

    first_converter = ImagePromptStyleConverter(converter_target=mock_target, filter_path=first)
    second_converter = ImagePromptStyleConverter(converter_target=mock_target, filter_path=second)

    assert first_converter.get_identifier().hash == second_converter.get_identifier().hash


def test_identifier_exposes_filter_content(mock_target, tmp_path) -> None:
    custom = tmp_path / "charcoal.yaml"
    _write_filter(custom, "Render as a charcoal sketch.", variation="close_up: tight framing")

    converter = ImagePromptStyleConverter(converter_target=mock_target, filter_path=custom)

    params = converter.get_identifier().params
    assert params["filter_name"] == "charcoal"
    assert params["style_instructions"] == "Render as a charcoal sketch."
    assert params["variations"] == {"close_up": "tight framing"}
    assert params["variation_order"] == ["close_up"]


def test_identifier_distinguishes_variations_of_same_filter(mock_target) -> None:
    bodycam = ImagePromptStyleConverter(
        converter_target=mock_target, filter_name="gritty_documentary", variation="bodycam_footage"
    )
    other = ImagePromptStyleConverter(converter_target=mock_target, filter_name="gritty_documentary")

    assert bodycam.get_identifier().params["variation"] == "bodycam_footage"
    assert other.get_identifier().params.get("variation") is None
    assert bodycam.get_identifier().hash != other.get_identifier().hash


def test_registry_accepts_same_named_custom_filters(mock_target, tmp_path, patch_central_database) -> None:
    """The issue's failure mode: registering converters whose filters share a
    file name raised a duplicate-name error."""
    from pyrit.registry import ConverterRegistry

    charcoal = tmp_path / "a" / "noir.yaml"
    charcoal.parent.mkdir()
    _write_filter(charcoal, "Shoot in black and white.")
    neon = tmp_path / "b" / "noir.yaml"
    neon.parent.mkdir()
    _write_filter(neon, "Render in neon colors.")

    charcoal_converter = ImagePromptStyleConverter(converter_target=mock_target, filter_path=charcoal)
    neon_converter = ImagePromptStyleConverter(converter_target=mock_target, filter_path=neon)

    registry = ConverterRegistry()
    registry.instances.register(charcoal_converter)
    registry.instances.register(neon_converter)


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    ("names", "variation"),
    [
        (("close_up", "CLOSE_UP"), "close_up"),
        (("close_up", " close_up "), "close_up"),
        (("close_up", "wide_shot"), None),
    ],
)
async def test_identifier_preserves_behavioral_variation_order_async(
    *,
    mock_target: MagicMock,
    tmp_path: Path,
    sqlite_instance: SQLiteMemory,
    names: tuple[str, str],
    variation: str | None,
) -> None:
    entries = list(zip(names, ("tight framing", "wide framing"), strict=True))
    converters = []
    for index, ordered_entries in enumerate((entries, entries[::-1])):
        path = tmp_path / str(index) / "noir.yaml"
        path.parent.mkdir()
        path.write_text(
            yaml.safe_dump(
                {"style_instructions": "Use charcoal.", "variations": dict(ordered_entries)},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        converters.append(
            ImagePromptStyleConverter(converter_target=mock_target, filter_path=path, variation=variation)
        )

    normalizer = PromptNormalizer()
    destination = MockPromptTarget()
    persisted_identifiers = []
    try:
        configure_random_seed(seed=42)
        for converter in converters:
            response = await normalizer.send_prompt_async(
                message=MessagePiece(role="user", original_value="a person walking").to_message(),
                target=destination,
                request_converter_configurations=[ConverterConfiguration(converters=[converter])],
            )
            requests = await sqlite_instance.get_message_pieces_async(
                conversation_id=response.get_piece().conversation_id, role="user"
            )
            assert len(requests) == 1
            persisted_identifiers.append(requests[0].converter_identifiers[0])
    finally:
        configure_random_seed(seed=None)

    system_prompts = [call.kwargs["system_prompt"] for call in mock_target.set_system_prompt_async.call_args_list]
    assert len(system_prompts) == 2
    assert system_prompts[0] != system_prompts[1]
    assert all("Use charcoal." in prompt for prompt in system_prompts)
    assert all(("tight framing" in prompt) != ("wide framing" in prompt) for prompt in system_prompts)

    identifiers = [converter.get_identifier() for converter in converters]
    assert identifiers[0].hash != identifiers[1].hash
    assert identifiers[0].unique_name != identifiers[1].unique_name
    assert [identifier.params["variation_order"] for identifier in identifiers] == [list(names), list(names[::-1])]
    assert persisted_identifiers == identifiers
    for identifier in identifiers:
        assert ConverterIdentifier.model_validate_json(identifier.model_dump_json()) == identifier

    registry = ConverterRegistry()
    for converter in converters:
        registry.instances.register(converter)
        assert registry.instances.get(converter.get_identifier().unique_name) is converter
    stored = await sqlite_instance.get_converter_identifiers_async(
        identifier_hashes=[identifier.hash for identifier in identifiers]
    )
    assert len(stored) == 2
    assert {identifier.hash: identifier.params for identifier in stored} == {
        identifier.hash: identifier.params for identifier in identifiers
    }
