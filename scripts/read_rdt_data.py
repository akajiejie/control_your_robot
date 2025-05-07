import h5py
import numpy as np
import cv2

def print_hdf5_group_info(group, indent=0):
    """
    递归打印HDF5 Group的详细信息。
    跳过 'instruction' 数据集。
    """
    for key in group.keys():
        if key == 'instruction':  # 跳过 'instruction'
            continue

        item = group[key]
        indent_str = ' ' * indent
        if isinstance(item, h5py.Group):
            print(f"{indent_str}Group: {key}")
            print_hdf5_group_info(item, indent + 2)  # 递归进入子Group
        elif isinstance(item, h5py.Dataset):
            print(f"{indent_str}Dataset: {key}")
            print(f"{indent_str}  Shape: {item.shape}, dtype: {item.dtype}")
            # print(f"{indent_str}  Shape: {item.shape}")

def save_video_from_bytes(img_bytes, video_path, fps=30, frame_size=(640, 480)):
    """
    解码二进制压缩图像数据并保存为本地视频。

    Args:
        img_bytes (bytes): 二进制压缩图像数据。
        video_path (str): 输出视频文件的路径。
        fps (int): 视频的帧率。
        frame_size (tuple): 视频的分辨率。
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 XVID 编码
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

        for i,img_byte in enumerate(img_bytes):
            nparr = np.frombuffer(img_byte, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 解码为彩色图像
            if img is not None:
                # 调整图像大小，以便与视频分辨率一致
                img_resized = cv2.resize(img, frame_size)
                # 将图像写入视频文件
                video_writer.write(img_resized)
            else:
                print("Image decoding failed.")
    except Exception as e:
        print(f"Error decoding image: {e}")
    finally:
        # 释放 VideoWriter 资源
        if 'video_writer' in locals():
            video_writer.release()

def display_image_from_bytes(img_bytes):
    """
    解码二进制压缩图像数据并显示图像。

    Args:
        img_bytes (bytes): 二进制压缩图像数据。
    """
    try:
        # 使用 OpenCV 解码压缩的图像数据
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 解码为彩色图像
        
        if img is not None:
            cv2.imshow("Decoded Image", img)
            cv2.waitKey(10)  # 确保图像刷新
        else:
            print("Image decoding failed.")
    except Exception as e:
        print(f"Error decoding image: {e}")


def print_hdf5_file_info_and_display_images(file_path):
    """
    读取HDF5文件中的所有数据，包括图像、状态、动作等，并展示图像。
    跳过 'instruction' 数据集，并且正确解析嵌套在 'observations' 中的数据。

    Args:
        file_path (str): HDF5文件路径。
    """
    with h5py.File(file_path, 'r') as f:
        # 打印文件结构
        print("HDF5 File Structure:")
        print_hdf5_group_info(f)  # 递归打印HDF5结构
        print("-" * 40)

        # 遍历所有数据集
        print(f.keys())
        for key in f.keys():
            # 跳过 instruction 数据集
            if key == 'instruction':
                continue

            dataset = f[key]
            # 如果数据集是 observations，解析内部的数据
            if key == 'observations':
                print("Parsing nested observations data:")
                # 获取嵌套数据
                if 'qpos' in dataset:
                    print(f"qpos shape: {dataset['qpos'].shape}")
                    print(f"Sample qpos data (first 5 elements): {dataset['qpos'][:5]}")
                if 'effort' in dataset:
                    print(f"effort shape: {dataset['effort'].shape}")
                    print(f"Sample effort data (first 5 elements): {dataset['effort'][:5]}")
                if 'qvel' in dataset:
                    print(f"qvel shape: {dataset['qvel'].shape}")
                    print(f"Sample qvel data (first 5 elements): {dataset['qvel'][:5]}")

                if 'images' in dataset:
                    # 解析图片数据
                    print("Parsing images in observations:")
                    for image_key in dataset['images']:
                        img_data = dataset['images'][image_key][:]
                        print(f"Displaying images from {image_key}:")
                        # 保存视频
                        # save_video_from_bytes(img_data, f"datasets/{image_key}.mp4", 10)
                        for i, img_bytes in enumerate(img_data):
                            continue
                            display_image_from_bytes(img_bytes)
            else:
                # 打印前5个数据样本
                print(f"Sample data (first 5 elements): {dataset[:5]}")

        cv2.destroyAllWindows()

# 使用示例：读取指定路径下的HDF5文件并打印内容和显示图像
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Show your rdt data.')
    parser.add_argument('episode_id', type=int,
                    help='your episode id, like episode_0.hdf5 id is 0')
    # 添加 data_path 参数，设置默认值
    parser.add_argument('data_path', type=str, nargs='?',default="datasets/RDT/",
                    help="your data dir like: datasets/RDT/")
    args = parser.parse_args()
    data_path = args.data_path
    i = args.episode_id
    file_path = f"datasets/RDT/episode_{i}.hdf5"
    print_hdf5_file_info_and_display_images(file_path)
