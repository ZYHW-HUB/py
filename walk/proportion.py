import cv2
import numpy as np

def calculate_color_percentage(image_path, target_color, tolerance):
    # 加载图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 定义颜色范围
    lower_color = np.clip(target_color - tolerance, 0, 255)
    upper_color = np.clip(target_color + tolerance, 0, 255)
    
    # 创建掩码
    mask = cv2.inRange(image, lower_color, upper_color)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # 统计掩码中非零像素的数量
    non_zero_pixels = cv2.countNonZero(mask)
    
    # 计算总面积
    total_pixels = image.shape[0] * image.shape[1]
    
    # 计算颜色区域的比例
    percentage = (non_zero_pixels / total_pixels) * 100
    
    return non_zero_pixels, percentage

# 定义目标颜色（例如红色）
target_color = np.array([35,142,107])
tolerance = 20  # 允许的偏差范围

# 计算特定颜色的像素数量及其占比
specific_color_pixels, specific_color_percentage = calculate_color_percentage('walk/deeplabv3-master/training_logs/model_eval_seq/berlin_000000_000019_pred.png', target_color, tolerance)
print(f"Pixels of the specific color: {specific_color_pixels}")
print(f"Percentage of the specific color: {specific_color_percentage:.2f}%")
# import cv2
# import numpy as np

# # 定义类别颜色映射
# color_map = {
#     0: [128, 64, 128],
#     1: [244, 35, 232],
#     2: [70, 70, 70],
#     3: [102, 102, 156],
#     4: [190, 153, 153],
#     5: [153, 153, 153],
#     6: [250, 170, 30],
#     7: [220, 220, 0],
#     8: [107, 142, 35],
#     9: [152, 251, 152],
#     10: [70, 130, 180],
#     11: [220, 20, 60],
#     12: [255, 0, 0],
#     13: [0, 0, 142],
#     14: [0, 0, 70],
#     15: [0, 60, 100],
#     16: [0, 80, 100],
#     17: [0, 0, 230],
#     18: [119, 11, 32],
#     19: [81, 0, 81]
# }

# # 初始化类别计数字典
# category_counts = {i: 0 for i in range(len(color_map))}

# def label_img_to_color(label_img):
#     """
#     将标签图像转换为带有颜色信息的图像。
#     :param label_img: 单通道的标签图像 (shape: (img_h, img_w))
#     :return: 彩色标签图像 (shape: (img_h, img_w, 3))
#     """
#     height, width = label_img.shape
#     color_img = np.zeros((height, width, 3), dtype=np.uint8)
    
#     for row in range(height):
#         for col in range(width):
#             class_id = label_img[row, col]
#             if class_id in color_map:
#                 color_img[row, col, :] = color_map[class_id]
    
#     return color_img

# def count_categories(pred_label_img, category_counts):
#     """
#     统计每个类别的像素数量。
#     :param pred_label_img: 预测的标签图像 (shape: (img_h, img_w))
#     :param category_counts: 类别计数字典
#     :return: 更新后的类别计数字典
#     """
#     for row in range(pred_label_img.shape[0]):
#         for col in range(pred_label_img.shape[1]):
#             class_id = pred_label_img[row, col]
#             if class_id in category_counts:
#                 category_counts[class_id] += 1
    
#     return category_counts

# def calculate_category_percentages(category_counts, total_pixels):
#     """
#     计算每个类别的占比。
#     :param category_counts: 类别计数字典
#     :param total_pixels: 总像素数量
#     :return: 类别占比字典
#     """
#     percentages = {}
#     for category, count in category_counts.items():
#         percentages[category] = (count / total_pixels) * 100
#     return percentages

# # 示例代码
# for i in range(pred_label_imgs.shape[0]):
#     pred_label_img = pred_label_imgs[i]  # (shape: (img_h, img_w))
#     img_id = img_ids[i]
#     img = imgs[i]  # (shape: (3, img_h, img_w))

#     img = img.data.cpu().numpy()
#     img = np.transpose(img, (1, 2, 0))  # (shape: (img_h, img_w, 3))
#     img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
#     img = img * 255.0
#     img = img.astype(np.uint8)

#     pred_label_img_color = label_img_to_color(pred_label_img)
#     overlayed_img = 0.35 * img + 0.65 * pred_label_img_color
#     overlayed_img = overlayed_img.astype(np.uint8)

#     # 统计每个类别的像素数量
#     category_counts = count_categories(pred_label_img, category_counts)

#     img_h = overlayed_img.shape[0]
#     img_w = overlayed_img.shape[1]

#     # 计算总面积
#     total_pixels = img_h * img_w

#     # 计算类别占比
#     category_percentages = calculate_category_percentages(category_counts, total_pixels)

#     # 打印结果
#     print("Category Counts:", category_counts)
#     print("Category Percentages:", category_percentages)