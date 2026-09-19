"""Live tests for 2.3 against the real chat server.

The headline capability of this project -- image understanding -- is NOT
available on this build. The chat server reports
`modalities: {vision: false, video: false, audio: false}` and answers any image
content block with HTTP 500 and the hint "you may need to provide the mmproj".
No multimodal projector is loaded, and SmolLM2-360M is not a vision model.

So this file does three honest things instead of faking a pass:

1. Pins the *unsupported* behaviour precisely, as positive assertions, so the
   day someone starts llama-server with --mmproj these tests fail and say the
   capability has arrived.
2. Verifies the project surfaces that failure rather than swallowing it -- a
   silently-dropped image would be far worse than a loud error.
3. Exercises everything that does NOT need vision against the real server:
   the text-only _generate path, the multimodal message envelope, and real
   images loaded from disk by the real ImageLoader.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from PIL import Image, ImageDraw

from extraction_chain import MultimodalExtractionChain, VisionChatModel
from loaders import ImageLoader


@pytest.fixture
def model(chat_base_url):
    m = VisionChatModel(base_url=chat_base_url, max_tokens=16)
    yield m
    m.close()


@pytest.fixture(scope="module")
def png_path(tmp_path_factory):
    """A real PNG on disk: a red square on white."""
    img = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(img).rectangle([8, 8, 56, 56], fill="red")
    path = tmp_path_factory.mktemp("images") / "square.png"
    img.save(path, format="PNG")
    return path


@pytest.fixture
def data_uri(png_path):
    """The same image as a data URI, produced by the project's own loader."""
    return ImageLoader().load_file(png_path).data_uri


# ---------------------------------------------------------------------------
# What this build actually supports
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_server_reports_no_vision_modality(chat_base_url):
    """/props is the authoritative capability check."""
    with urllib.request.urlopen(f"{chat_base_url}/props", timeout=10) as resp:
        props = json.loads(resp.read().decode())
    modalities = props["modalities"]
    assert modalities["vision"] is False
    assert modalities["audio"] is False
    # If this ever flips to True, the xfails below should be removed.


@pytest.mark.live
def test_image_input_is_rejected_with_an_mmproj_hint(chat_base_url, data_uri):
    """The exact server response for an image block, pinned.

    Note the status is 500 / "server_error" -- unlike the missing-embeddings
    and missing-reranking capabilities, which answer 501 /
    "not_supported_error". Recorded in drift-rag.md.
    """
    client = httpx.Client(base_url=chat_base_url, timeout=60.0)
    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "vision",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What colour is the shape?"},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                "max_tokens": 16,
            },
        )
        assert resp.status_code == 500
        err = resp.json()["error"]
        assert "image input is not supported" in err["message"]
        assert "mmproj" in err["message"]
    finally:
        client.close()


@pytest.mark.live
def test_invoke_with_images_surfaces_the_error(model, data_uri):
    """The project must not silently drop the image and answer from text alone.

    This is the important behavioural assertion in this file: a silent
    image-drop would produce plausible-looking output with the image ignored,
    which is a far more dangerous failure than an exception.
    """
    with pytest.raises(httpx.HTTPStatusError):
        model.invoke_with_images(
            prompt="What colour is the shape?", images=[data_uri]
        )


@pytest.mark.live
@pytest.mark.xfail(
    reason="vision unsupported: no --mmproj on this build, and SmolLM2-360M "
    "is not a vision model",
    strict=True,
    raises=httpx.HTTPStatusError,
)
def test_extraction_chain_extract(chat_base_url, data_uri):
    """Would be the project's primary use case. Cannot run here."""
    chain = MultimodalExtractionChain(base_url=chat_base_url, max_tokens=32)
    try:
        chain.extract(data_uri, prompt="What colour is the shape?")
    finally:
        chain.close()


@pytest.mark.live
@pytest.mark.xfail(
    reason="vision unsupported: no --mmproj on this build",
    strict=True,
    raises=httpx.HTTPStatusError,
)
def test_extraction_chain_extract_batch(chat_base_url, data_uri):
    chain = MultimodalExtractionChain(base_url=chat_base_url, max_tokens=32)
    try:
        chain.extract_batch([data_uri, data_uri])
    finally:
        chain.close()


# ---------------------------------------------------------------------------
# The non-vision paths, which do work against the real server
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_text_only_generate_works(model):
    """VisionChatModel._generate is a plain text path and needs no projector."""
    msg = model.invoke([HumanMessage(content="Say hi")])
    assert isinstance(msg, AIMessage)
    assert isinstance(msg.content, str)
    assert msg.content.strip()


@pytest.mark.live
def test_text_only_generate_with_system_message(model):
    msg = model.invoke(
        [
            SystemMessage(content="You are terse."),
            HumanMessage(content="Say hi"),
        ]
    )
    assert msg.content.strip()


@pytest.mark.live
def test_multimodal_envelope_is_accepted_without_images(model):
    """A content-parts list with only a text block is fine.

    Proves the request envelope itself is valid OpenAI-multimodal shape and
    that only the image block is what the server refuses -- so the failure
    above really is about the missing projector, not a malformed body.
    """
    out = model.invoke_with_images(prompt="Say hi", images=[])
    assert isinstance(out, str)
    assert out.strip()


@pytest.mark.live
def test_message_builder_shape(model, data_uri):
    """_build_multimodal_messages produces the shape the server expects."""
    messages = model._build_multimodal_messages(
        "describe", [data_uri], system_prompt="be terse"
    )
    assert messages[0] == {"role": "system", "content": "be terse"}
    parts = messages[1]["content"]
    assert parts[0] == {"type": "text", "text": "describe"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Real images from disk, via the project's loader
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_image_loader_produces_a_usable_data_uri(png_path):
    loaded = ImageLoader().load_file(png_path)
    assert loaded.media_type == "image/png"
    assert loaded.width == 64 and loaded.height == 64
    assert loaded.file_size_bytes > 0
    header, _, payload = loaded.data_uri.partition(",")
    assert header == "data:image/png;base64"
    # The payload must be real, decodable PNG bytes.
    decoded = base64.b64decode(payload)
    assert Image.open(io.BytesIO(decoded)).size == (64, 64)


@pytest.mark.live
def test_image_loader_resize_is_applied(png_path):
    loaded = ImageLoader(max_width=32, max_height=32).load_file(png_path)
    assert loaded.width <= 32 and loaded.height <= 32
    decoded = base64.b64decode(loaded.data_uri.partition(",")[2])
    assert Image.open(io.BytesIO(decoded)).size == (loaded.width, loaded.height)
