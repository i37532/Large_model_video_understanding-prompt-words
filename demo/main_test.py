import threading
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware
import business

# 创建 FastAPI 应用实例
app = FastAPI()

# 定义允许的跨域请求来源列表
origins = [
    # 本地部署无所谓
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

# model_path = "/qw/Qwen2.5-VL-7B-Instruct"
# forward_path = "/data/lihui"
forward_path = ""
lock = threading.Lock()
connect_clients = set()


def async_task(json_data):
    # 读取请求体中的原始数据
    # 读取请求体中的原始数据

    file_path = forward_path + "/qw/"
    video_url = file_path + json_data["video_name"]
    # 校验文件是否存在
    # if not os.path.exists(video_url):
    #     raise HTTPException(status_code=404, detail="Video not found")
    # 校验是否为视频文件 ============================================
    event = json_data["event"]
    # sys_prompt 系统提示词 ==============================================
    if "sys_prompt" in json_data and json_data["sys_prompt"]:
        sys_prompt = json_data["sys_prompt"]
    else:
        sys_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    # 温度设置 =============================================
    if "temperature" in json_data and json_data["temperature"]:
        temperature = json_data["temperature"]
    else:
        temperature = 1.0
    # 温度设置 =============================================
    background = ""
    if "background" in json_data and json_data["background"]:
        background = json_data["background"]
    # 删除上传的视频文件
    # os.remove(video_url)
    # 在这里上锁
    is_locked = lock.locked()
    if lock.locked():
        return {"message": "当前其他任务正在完成"}
    else:
        if lock.acquire(blocking=False):
            thread = threading.Thread(target=business.test_time, args=(lock, connect_clients))
            thread.start()
    return {"message": "成功"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connect_clients.add(websocket)
    try:
        while True:
            result = await websocket.receive_text()
            print("收到消息:",result)
    except WebSocketDisconnect:
        connect_clients.remove(websocket)


# 定义一个 POST 请求处理函数，接收一个 Item 模型的请求体
@app.post("/predict")
async def handle_post(request: Request):
    raw_data = await request.body()
    # 将原始数据解析为 JSON 对象
    json_data = json.loads(raw_data)
    result = async_task(json_data)
    return {"message": result}


if __name__ == "__main__":
    uvicorn.run("main_test:app", host="0.0.0.0", port=7532, workers=1)
