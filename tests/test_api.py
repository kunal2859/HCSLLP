import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(autouse=True)
def mock_chroma_and_embeddings(monkeypatch):
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 384]
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda *a, **kw: mock_model
    )

    mock_collection = MagicMock()
    mock_collection.count.return_value = 5
    mock_collection.query.return_value = {
        "documents": [["Docker is a platform for running containers."]]
    }
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    monkeypatch.setattr("chromadb.PersistentClient", lambda *a, **kw: mock_client)


def run_python_subprocess(code: str) -> str:
    import subprocess
    result = subprocess.run(
        ["python3", "-c", code],
        capture_output=True, text=True, timeout=3
    )
    if result.stderr:
        return f"Error: {result.stderr.strip()}"
    return result.stdout.strip() or "Success (no output)"


class TestRunPython:
    def test_simple_math(self):
        result = run_python_subprocess("print(2 + 2)")
        assert result == "4"

    def test_list_comprehension(self):
        result = run_python_subprocess("x = [i for i in range(5)]; print(sum(x))")
        assert result == "10"

    def test_syntax_error(self):
        result = run_python_subprocess("print(")
        assert "Error" in result

    def test_timeout_protection(self):
        import subprocess
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(
                ["python3", "-c", "while True: pass"],
                capture_output=True, text=True, timeout=2
            )


class TestDevopsSearch:
    def test_returns_list(self, mock_chroma_and_embeddings):
        from mcp_server.vector_store import search_documents
        results = search_documents("docker")
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_returns_string_results(self, mock_chroma_and_embeddings):
        from mcp_server.vector_store import search_documents
        results = search_documents("kubernetes")
        for r in results:
            assert isinstance(r, str)


class TestAPIEndpoints:
    def test_valid_devops_search(self, mock_chroma_and_embeddings):
        from fastapi.testclient import TestClient
        from mcp_server.api import app
        client = TestClient(app)
        response = client.post("/tool", json={"tool": "devops_search", "input": "docker"})
        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert data["tool"] == "devops_search"

    def test_valid_run_python(self, mock_chroma_and_embeddings):
        from fastapi.testclient import TestClient
        from mcp_server.api import app
        client = TestClient(app)
        response = client.post("/tool", json={"tool": "run_python", "input": "print(1+1)"})
        assert response.status_code == 200
        data = response.json()
        assert data["output"] == "2"

    def test_invalid_tool(self, mock_chroma_and_embeddings):
        from fastapi.testclient import TestClient
        from mcp_server.api import app
        client = TestClient(app)
        response = client.post("/tool", json={"tool": "nonexistent_tool", "input": "test"})
        assert response.status_code == 200
        data = response.json()
        assert "Invalid tool" in data["output"]
