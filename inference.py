import time
import logging
import argparse
import cv2
import numpy as np
import mindspore
from mindspore import context, ops

from dct_generator import Generator

# 配置日志记录
logging.basicConfig(filename="./dct_net.log",
                    filemode='a',
                    level=logging.INFO,  # 设置日志级别
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 设置日志格式

def camera(network, args):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("无法打开摄像头")
        exit()

    while True:
        begin = time.time()
        ret, img = cap.read()
        if not ret:
            logging.error("无法读取视频流")
            time.sleep(10)
            continue

        img_h, img_w, _ = img.shape
        logging.debug("original image size:({}, {})".format(img_h, img_w))
        origin = img.copy()
        if args.speed_first == "yes":
            img = cv2.resize(img, (320, 240))
        else:
            img = cv2.resize(img, (img_w // 8 * 8, img_h // 8 * 8))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img[...,::-1] / 255.0 - 0.5) * 2 
        img = img.transpose(2, 0, 1)[np.newaxis,:].astype(np.float32)
        inp = mindspore.Tensor(img)
        xg = network(inp)[0]
        xg = (xg + 1) * 0.5
        xg = ops.clamp(xg * 255 + 0.5, 0, 255)
        xg = xg.permute(1, 2, 0).asnumpy()[...,::-1]
        xg = cv2.cvtColor(xg, cv2.COLOR_RGB2BGR)
        xg = xg.astype(np.uint8)
        xg = cv2.resize(xg, (origin.shape[1], origin.shape[0]))
        result = np.hstack((origin, xg))

        text_left = "origin image:({}x{})".format(img_h, img_w)
        text_right = "processed image:({}x{})".format(img_h, img_w)
        # 设置字体大小和颜色
        font_scale = 0.8  # 字体大小
        color = (255, 255, 255)  # 白色文字 (B, G, R)
        thickness = 2  # 文字厚度
        # 设置字体类型
        font = cv2.FONT_HERSHEY_SIMPLEX
        (_, text_height), _ = cv2.getTextSize(text_left, font, font_scale, thickness)

        # 计算文本的起始位置
        x = 10
        # Y轴的位置设置为文本高度 + 上边距
        y = text_height + 20  # 20 像素的上边距

        # 在图像上绘制文本
        cv2.putText(result, text_left, (x, y), font, font_scale, color, thickness)
        cv2.putText(result, text_right, (x + img_w, y), font, font_scale, color, thickness)


        # 显示图像
        cv2.imshow('MindSpore Application [DCT-Net] on OrangePi', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
            break
        end = time.time()
        logging.debug('process image cost time:{}'.format(end - begin))


def proc_one_img(network, args):
    img = cv2.imread(args.img_path)
    begin = time.time()
    img_h, img_w, _ = img.shape
    logging.info("original image size:({}, {})".format(img_h, img_w))
    img = cv2.resize(img, (img_w // 8 * 8, img_h // 8 * 8))
    img = (img[...,::-1] / 255.0 - 0.5) * 2
    img = img.transpose(2, 0, 1)[np.newaxis,:].astype(np.float32)
    inp = mindspore.Tensor(img)
    xg = network(inp)[0]
    xg = (xg + 1) * 0.5
    xg = ops.clamp(xg*255+0.5,0,255)
    xg = xg.permute(1,2,0).asnumpy()[...,::-1]
    cv2.imwrite(args.output_path, xg)
    end = time.time()
    logging.info('process image cost time:{}, output image path:{}'.format(end-begin, args.output_path))
    

if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description="DCT-Net Inference")
    # 添加参数
    parser.add_argument("--run_mode", type=str, default="GRAPH_MODE", help="运行模式：'GRAPH_MODE' or 'PYNATIVE'")
    parser.add_argument("--device_type", type=str, default="Ascend", help="设备类型：'CPU' or 'Ascend'")
    parser.add_argument("--device_id", type=int, default=0, help="设备ID")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/dct-net.ckpt", help="模型文件路径")
    parser.add_argument("--camera", type=str, default="yes", help="yes:使用摄像头实时推理，no:使用本地图片推理")
    parser.add_argument("--speed_first", type=str, default="yes", help="yes:优先考虑速度，no:优先考虑质量")
    parser.add_argument("--img_path", type=str, default="./images/gdg.png", help="源图像文件路径")
    parser.add_argument("--output_path", type=str, default="./images/output.png", help="输出图像文件路径")
    
    # 解析参数
    args = parser.parse_args()

    context.set_context(
        mode=context.GRAPH_MODE if args.run_mode == "GRAPH_MODE" else context.PYNATIVE_MODE,
        device_target=args.device_type,
        device_id=args.device_id,
        jit_config={"jit_level": "O2"}
    )

    logging.info("run_mode:{}\ndevice_type:{}\ndevice_id:{}\nckpt_path:{}\ncamera:{}\nspeed_first:{}\nimg_path:{}\noutput_path:{}\n".format(\
        args.run_mode, args.device_type, args.device_id, args.ckpt_path, args.camera, args.speed_first, args.img_path, args.output_path))

    # 加载模型
    network = Generator(img_channels=3)
    mindspore.load_checkpoint(args.ckpt_path, network)
    network.set_train(mode=False)
    if args.camera == "yes":
        logging.info('*' * 50 + "use camera" + '*' * 50)
        camera(network, args)
    else:
        logging.info('*' * 50 + "proc local image" + '*' * 50)
        proc_one_img(network, args)
