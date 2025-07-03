import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import json


# 模拟一个耗时的同步方法
async def sync_long_running_task(connect_clients):
    for client in connect_clients:
        try:
            print("开始向客户端发送信息")
            await client.send_json(json.dumps({"data": 185, "进度": 11 / 85, "描述": "你好"},ensure_ascii=False))
        except Exception as e:
            print(f"Error sending message: {e}")


def test_time(lock, connect_clients):
    try:
        print("同步耗时任务开始")
        time.sleep(2)  # 模拟 2 秒的耗时操作
        # 创建一个事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        def run_async_task():
            nonlocal loop
            try:
                result = loop.run_until_complete(sync_long_running_task(connect_clients))
                print(f"Received response: {result}")
            finally:
                loop.close()
        thread = threading.Thread(target=run_async_task)
        thread.start()
        thread.join()
    finally:
        if lock.locked():
            lock.release()
