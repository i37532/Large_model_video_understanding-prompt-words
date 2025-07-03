import queue

import cv2
import threading
import os
from queue import Queue
import time
import multiprocessing

# 每秒抽取的帧数
frame_rate = 1

# 并发线程数
num_threads = multiprocessing.cpu_count() // 2
print(multiprocessing.cpu_count())
# 启动多个线程读取视频帧
threads = []
# 用于存储帧的队列
frames_queue = Queue(maxsize=500)
all_pic_path = "all_pic"


# 从视频中读取指定帧
def frame_extractor(start_frame, end_frame, video_path):
    cap = cv2.VideoCapture(video_path)
    # 获取视频的帧率
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    # 计算每隔多少帧抽取一次，因为每秒抽两帧，所以间隔帧数为帧率除以2
    frame_interval = int(fps / frame_rate)
    # 设置视频位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # 从开始帧到结束帧
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:  # 每秒抽取一帧
            frames_queue.put((frame_num, frame))  # 将帧和其索引放入队列
    cap.release()


# 保存帧的函数
def frame_saver(output_dir):
    while True:
        # print(f"当前队列待处理帧数{frames_queue.qsize()}")
        if all(not thread.is_alive() for thread in threads) and frames_queue.empty():
            break
        else:
            try:
                frame_data = frames_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # frames_queue.empty()检查是否为空

            # if frame_data is None:  # 接收到结束信号
            #     break
            frame_num, frame = frame_data
            file_name = f'frame_{frame_num}.jpg'
            frame_filename = os.path.join(output_dir, file_name)
            cv2.imwrite(frame_filename, frame)
            frames_queue.task_done()


def read_file(output_dir):
    # 获取文件夹下的所有文件和文件夹名称
    file_list = os.listdir(output_dir)
    # 过滤出文件（去除文件夹）
    files = [f for f in file_list if os.path.isfile(os.path.join(output_dir, f))]
    files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    return files


def extract_and_save(video_path):
    output_dir = all_pic_path + "/" + video_path.split("/")[-1].replace(".", "") + str(time.time()).replace(".", "")
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    # 获取视频总帧数
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # 计算每个线程处理的帧范围
    thread_ranges = [(i * (total_frames // num_threads), (i + 1) * (total_frames // num_threads)) for i in
                     range(num_threads)]
    # 处理最后一个线程的剩余帧
    thread_ranges[-1] = (thread_ranges[-1][0], total_frames)

    for start_frame, end_frame in thread_ranges:
        thread = threading.Thread(target=frame_extractor, args=(start_frame, end_frame, video_path))
        thread.start()
        threads.append(thread)

    # 启动帧保存线程 3个处理帧
    save_threads = []
    for i in range(multiprocessing.cpu_count() - 2):
        saver_thread = threading.Thread(target=frame_saver, args=(output_dir,))
        saver_thread.start()
        save_threads.append(saver_thread)
    # 等待所有提取线程完成
    for thread in threads:
        thread.join()

    for thread in save_threads:
        thread.join()
    print(f"总耗时{time.time() - start_time}")
    print("视频帧抽取完成！")
    # 排序
    return read_file(output_dir), output_dir


if __name__ == '__main__':
    video_path = "../湿敏仓-3_20220816170641-20220816171341_1.mp4"
    extract_and_save(video_path)
