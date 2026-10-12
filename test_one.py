import requests
import time

start = time.time()
resp = requests.post("http://localhost:8000/api/decide", json={"question": "番茄炒蛋怎么做？"}, timeout=120)
elapsed = time.time() - start

print(f"耗时: {elapsed:.1f} 秒")
print(f"状态码: {resp.status_code}")
print(f"返回: {resp.json()}")
