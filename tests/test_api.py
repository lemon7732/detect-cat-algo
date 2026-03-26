import pytest


fastapi = pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from io import BytesIO

import anyio
import httpx
from PIL import Image

from cat_rescue_ai.api.app import create_app
from cat_rescue_ai.api.schemas import create_schema_namespace


async def _request(app, method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _run_request(app, method: str, path: str, **kwargs):
    return anyio.run(lambda: _request(app, method, path, **kwargs))


def test_schema_namespace_contains_identify_schema():
    schemas = create_schema_namespace()
    assert "IdentifyResponse" in schemas


def test_create_app_without_tensorflow_runtime():
    app = create_app("configs/api.yaml")
    assert app.title == "Campus Stray Cat Recognition API"


def test_health_endpoint():
    app = create_app("configs/api.yaml")
    response = _run_request(app, "GET", "/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_detect_endpoint_returns_no_face_for_blank_image():
    app = create_app("configs/api.yaml")

    image = Image.new("RGB", (128, 128), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = _run_request(
        app,
        "POST",
        "/detect/cat-face",
        files={"file": ("blank.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["faces"] == []
    assert payload["primary_bbox"] is None
