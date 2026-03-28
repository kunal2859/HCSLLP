import requests # type: ignore

BASE_URL = "http://127.0.0.1:8000/tool"

def test_search():
    print("\n--- Testing Search ---")
    res = requests.post(BASE_URL, json={
        "tool": "devops_search",
        "input": "docker"
    })
    print(res.json())

def test_python_valid():
    print("\n--- Testing Python (print) ---")
    res = requests.post(BASE_URL, json={
        "tool": "run_python",
        "input": "print(2+2)"
    })
    print(res.json())

def test_python_logic():
    print("\n--- Testing Python (logic) ---")
    res = requests.post(BASE_URL, json={
        "tool": "run_python",
        "input": "x = [i for i in range(5)]; print(sum(x))"
    })
    print(res.json())

def test_invalid():
    print("\n--- Testing Invalid Tool ---")
    res = requests.post(BASE_URL, json={
        "tool": "unknown_tool",
        "input": "test"
    })
    print(res.json())

if __name__ == "__main__":
    test_search()
    test_python_valid()
    test_python_logic()
    test_invalid()