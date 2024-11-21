import os

from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

def enhance_image(image, brightness_factor, contrast_factor):
    # 打开彩色图像
    # image = Image.open(image_path)

    # 将图像转换为NumPy数组
    image_array = np.array(image)

    # 将图像数组从0-255范围转换为0-1范围
    image_array = image_array / 255.0

    # 提高亮度
    enhanced_image_array = image_array * brightness_factor

    # 提高对比度
    enhanced_image_array = (enhanced_image_array - 0.5) * contrast_factor + 0.5

    # 将图像数组从0-1范围转换回0-255范围
    enhanced_image_array = np.clip(enhanced_image_array * 255.0, 0, 255).astype(np.uint8)

    # 创建增强后的图像对象
    enhanced_image = Image.fromarray(enhanced_image_array)

    # 保存增强后的图像
    # enhanced_image.save('enhanced_image_test.png')

    return enhanced_image


def convolve_image_rgb(image_path):
    # 打开彩色图像
    image = Image.open(image_path)

    # 转换为Tensor对象
    tensor_image = torch.Tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()

    # 定义卷积核
    kernel_x = torch.tensor([[-1, 1], [-2, 2]], dtype=torch.float32).view(1, 1, 2, 2)
    kernel_y = torch.tensor([[2, 1], [-2, -1]], dtype=torch.float32).view(1, 1, 2, 2)

    # 获取通道数
    num_channels = tensor_image.size(1)

    # 初始化输出张量
    output = torch.zeros_like(tensor_image)

    # 针对每个通道进行卷积运算
    for channel in range(num_channels):
        # 当前通道的图像数据
        channel_image = tensor_image[:, channel:channel + 1, :, :]

        # 填充图像
        padded_image = F.pad(channel_image, (1, 0, 1, 0), mode='constant')

        # 卷积运算
        conv_x = F.conv2d(padded_image, kernel_x, stride=1, padding=0)
        conv_y = F.conv2d(padded_image, kernel_y, stride=1, padding=0)

        # 计算绝对值之和的平方根
        output_channel = torch.sqrt(torch.pow(conv_x.abs(), 2) + torch.pow(conv_y.abs(), 2))

        # 将当前通道的输出存储到输出张量中
        output[:, channel:channel + 1, :, :] = output_channel

    # 将输出Tensor对象转换为彩色图像
    output = output.squeeze().permute(1, 2, 0).numpy()
    output = output.astype(np.uint8)
    output_image = Image.fromarray(output)

    # 保存彩色图像
    # output_image.save('output_image_001.png')

    return output




def convolve_image(image_path):
    # 打开图像并转化为灰度图
    image = Image.open(image_path).convert('L')

    # 转化为Tensor对象
    tensor_image = torch.Tensor(list(image.getdata())).view(1, 1, image.size[1], image.size[0])

    # 定义卷积核
    kernel_x = torch.tensor([[-1, 1], [-2, 2]], dtype=torch.float32).view(1, 1, 2, 2)
    kernel_y = torch.tensor([[2, 1], [-2, -1]], dtype=torch.float32).view(1, 1, 2, 2)

    # 卷积运算
    conv_x = F.conv2d(tensor_image, kernel_x, stride=1, padding=0)
    conv_y = F.conv2d(tensor_image, kernel_y, stride=1, padding=0)

    # 计算绝对值之和的平方根
    output = torch.sqrt(torch.pow(conv_x.abs(), 2) + torch.pow(conv_y.abs(), 2))

    # 将输出的Tensor对象转化为灰度图
    output_image = Image.fromarray(output.squeeze().numpy(), 'L')


    # 保存灰度图
    # output_image.save('tile001.png')

    return output


if __name__ == "__main__":
    # image_path = r"E:\crack_detect\img\002.jpg"
    image_path = "output_image_001.png"
    input_folder = r"E:\crack_detect\img\test\enhance_out"
    output_folder = r"E:\crack_detect\img\test\enhance_cvo_gray"
    # 获取输入文件夹中的所有文件列表
    file_list = os.listdir(input_folder)
    # 遍历文件列表
    for file_name in file_list:
        # 构建完整的文件路径
        image_path = os.path.join(input_folder, file_name)
        cvo_out_img = convolve_image_rgb(image_path)
        # cvo_out_img = Image.open(image_path)
        # 将图像转换为灰度图
        # 将 NumPy 数组转为 PIL Image 对象
        cvo_out_img = Image.fromarray(cvo_out_img)
        gray_image = cvo_out_img.convert('L')
        # out = enhance_image(cvo_out_img,1,3)
        output_path = os.path.join(output_folder, file_name)
        gray_image.save(output_path)



    # convolve_image(image_path)
    # convolve_image_rgb(image_path)
    # enhance_image(image_path,3,3)
