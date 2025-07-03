import json
import time
import torch
from qwen_vl_utils import process_vision_info
import asyncio
import threading
import os
import base64
from PIL import Image
from io import BytesIO

batch_size = 10


def batch_group(lst, size):
    result = []
    for i in range(0, len(lst), size):
        result.append(lst[i:i + size])
    return result


# 删除文件及文件夹
def delete_dir(path):
    import shutil
    shutil.rmtree(path)


def delete_pic(pic_list):
    for pic in pic_list:
        os.remove(pic)


def compress_and_convert_to_base64(image_path, quality=50):
    try:
        # 打开图片
        with Image.open(image_path) as img:
            # 创建一个内存中的字节流对象
            buffer = BytesIO()
            # 压缩图片并保存到字节流中
            img.save(buffer, format=img.format, quality=quality)
            # 获取字节流中的图片数据
            compressed_image_data = buffer.getvalue()
            # 对压缩后的图片数据进行 Base64 编码
            encoded_image = base64.b64encode(compressed_image_data)
            # 将编码后的字节对象转换为字符串
            base64_string = encoded_image.decode('utf-8')
            return base64_string
    except FileNotFoundError:
        print(f"文件 {image_path} 未找到。")
    except Exception as e:
        print(f"发生错误: {e}")


# 模拟一个耗时的同步方法
async def sync_long_running_task(connect_clients, result):
    for client in connect_clients:
        try:
            print("开始向客户端发送信息")
            await client.send_json(result)
        except Exception as e:
            print(f"Error sending message: {e}")


def send_message(connect_clients, result):
    # 创建一个事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def run_async_task(connect_clients_, result_):
        nonlocal loop
        try:
            loop.run_until_complete(sync_long_running_task(connect_clients_, result_))
        finally:
            loop.close()

    thread = threading.Thread(target=run_async_task, args=(connect_clients, result))
    thread.start()
    thread.join()


def predict(processor, model, file_list, output_dir, fps, connect_clients, item):
    # 开始时间
    count = 0
    start_time = time.time()
    batch_s = batch_group(file_list, batch_size)
    event = item.event
    model_output = []
    need_deleted_pic = []
    for batch in batch_s:
        messages_s = []
        for url in batch:
            messages = [
                {
                    "role": "system",
                    # ----------------------------------------------------------------------------------
                    "content": """根据用户的输入提示词,通过当前图片的背景来源,你需要仔细分析这张图片\
                               并且以json格式输出,一张图片可能存在多个事件,请你按照指定格式输出.
                               [{"event": // 事件名称
                                "explain": //事件说明,给出详细理由
                                "match": //符合程度,范围0~1.0,0代表不符合,0.5代表可能符合,1代表符合}]""",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "image": "./" + output_dir + "/" + url},
                        {
                            "type": "text",
                            # -------------------------------------------------------------------------------------
                            # 当前图片拍摄于家庭监控,你首先需要针对当前场景,分析是否发生了人身安全问题,简单扼要的指出,并且给出一个0~1的危险程度,"
                            # "0完全没有危险,0.3是可能存在危险,0.5轻危险,0.8是比较危险,1重大危险;识别结果字数30字以内

                            "text": f"背景来源:{item.background},请你仔细分析,是否存在{str(event)}"
                        }
                    ]
                }
            ]
            messages_s.append(messages)
        # Preparation for inference
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_s]
        image_inputs, video_inputs = process_vision_info(messages_s)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for index in range(len(output_text)):
            result = output_text[index]
            print(result)
            pic_name = batch[index]
            pic_url = ""
            s = result.strip().replace("```json", "").replace("```", "").strip()
            # 容器内地址
            container_url = os.path.dirname(os.path.abspath(__file__)) + "/" + output_dir + "/" + pic_name

            try:
                list_data = [data for data in json.loads(s) if data["match"] >= 0.8]

                for data in list_data:
                    match = data["match"]
                    print("match的类型:", type(match))
                if len(list_data) > 0:
                    # 服务器路径 + 容器内路径 + 文件路径
                    pic_url = compress_and_convert_to_base64(container_url, 50)
                else:
                    # 当前图片需要删除
                    need_deleted_pic.append(container_url)
            except Exception as e:
                print(f"当前大模型输出解析失败: {e}", f"  大模型输出:{result}")
                continue
            current = round(int(pic_name.split(".")[0].split("_")[-1]) / fps)
            # 判断是否应该保存当前该文件

            # 发消息前端
            count += 1
            result = json.dumps({
                "time": current,
                "video_name": item.video_name,
                "uuid": item.uuid,
                "text": list_data,
                "progress_bar": round(count / len(file_list), 3),
                "pic_url": pic_url}, ensure_ascii=False)
            send_message(connect_clients, result)

            model_output.append(result)
        del inputs, generated_ids_trimmed
    # 清空torch缓存
    torch.cuda.empty_cache()
    # 结束时间
    print(f"分析{len(file_list)}张图片,当前耗时{time.time() - start_time}秒")
    # 写入到本地
    # 打开文件以写入模式
    file_path = "model_output.txt"
    # 打开文件并写入 JSON 数据
    with open(file_path, 'w', encoding='utf-8') as file:
        # 使用 json.dump() 方法将数据写入文件
        # indent=4 参数用于格式化 JSON 数据，使其更易读
        json.dump(model_output, file, indent=4, ensure_ascii=False)
    print(f"数据已成功写入 {file_path}")
    delete_dir(output_dir)
    # 删除不需要的文件
    # if need_deleted_pic:
    #     print(f"当前一共{len(need_deleted_pic)}张图片需要删除")
    #     delete_pic(need_deleted_pic)
    #     print("图片删除完成")
