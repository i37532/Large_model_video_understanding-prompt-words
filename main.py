from fastapi import FastAPI, Request, HTTPException, WebSocketDisconnect
import uvicorn
import json
from starlette.websockets import WebSocket
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import os
from fastapi.middleware.cors import CORSMiddleware
import qwen_vl_demo_pic_much, 多线程抽帧_demo5
import cv2
import threading
from pydantic import BaseModel

# 创建 FastAPI 应用实例
app = FastAPI()
lock = threading.Lock()
connect_clients = set()
# 定义允许的跨域请求来源列表
origins = [
    # 本地部署无所谓
    "http://192.168.111.199",
    "*",
    # 可以根据需要添加更多允许的域名
]
# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的来源
    allow_credentials=True,  # 允许携带凭证（如 cookies）
    allow_methods=["*"],  # 允许的 HTTP 方法
    allow_headers=["*"],  # 允许的 HTTP 请求头
)

# forward_path = "/data/lihui"
forward_path = ""
model_path = forward_path + "/qw/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)


def deal_video(lock_, item):
    try:
        video_url = item.file_forward_path + item.video_name
        # 校验文件是否存在
        if not os.path.exists(video_url):
            raise HTTPException(status_code=404, detail="Video not found")
        file_list, output_dir = 多线程抽帧_demo5.extract_and_save(video_url)
        cap = cv2.VideoCapture(video_url)
        # 获取视频的帧率
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        # 分析代码
        qwen_vl_demo_pic_much.predict(processor, model, file_list, output_dir, fps, connect_clients, item)
        # 删除上传的视频文件
        os.remove(video_url)
    finally:
        if lock_.locked():
            lock_.release()


def sync_task(item):
    is_locked = lock.locked()
    if is_locked:
        return {"message": "当前其他任务正在执行", "code": 500}
    else:
        if lock.acquire(blocking=False):
            thread = threading.Thread(target=deal_video, args=(lock, item))
            thread.start()
    return {"message": "成功", "code": 200}


class Item(BaseModel):
    video_name: str
    event: list
    background: str
    file_forward_path: str = "/qw/"
    uuid:str


# 这里要求一次只能处理一个请求
# 主接口
@app.post("/predict")
async def handle_post(item: Item):
    return sync_task(item)


@app.get("/stop")
async def handle_post():
    os.system("uvicorn main:app --host 0.0.0.0 --port 7532")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connect_clients.add(websocket)
    try:
        while True:
            result = await websocket.receive_text()
            print("收到消息:", result)
    except Exception as e:
        print(f"Error receive message: {e}")
    finally:
        connect_clients.remove(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7532)
