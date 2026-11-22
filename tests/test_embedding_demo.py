import io
import json

import pytest

from rag_lab import embedding_demo as demo


def test_batch_preserves_input_order(monkeypatch):
    def respond(request, timeout):
        assert json.loads(request.data)["input"] == ["coffee", "shampoo"]
        assert request.get_method() == "POST"
        return io.BytesIO(b'{"data":[{"index":1,"embedding":[0,1]},{"index":0,"embedding":[1,0]}]}')

    monkeypatch.setattr(demo, "urlopen", respond)
    assert demo.embed_texts(["coffee", "shampoo"], "example-model", "test") == [[1, 0], [0, 1]]


def test_missing_result_is_rejected(monkeypatch):
    monkeypatch.setattr(demo, "urlopen", lambda *a, **kw: io.BytesIO(b'{"data":[]}'))
    with pytest.raises(ValueError, match="indices"):
        demo.embed_texts(["coffee"], "example-model", "test")


def test_blank_input_is_rejected_before_request():
    with pytest.raises(ValueError, match="nonblank"):
        demo.embed_texts(["  "], "example-model", "test")
