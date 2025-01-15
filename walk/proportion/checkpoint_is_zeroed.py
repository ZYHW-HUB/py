import os
checkpoint_file = 'py/walk/proportion/proportion_checkpoint.txt'
# 打印检查点文件的位置
print(f"Checkpoint file location: {os.path.abspath(checkpoint_file)}")

# 检查检查点文件是否存在
if os.path.exists(checkpoint_file):
    print("Checkpoint file exists.")
else:
    print("Checkpoint file does not exist.")
def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return 0

def save_checkpoint(index):
    with open(checkpoint_file, 'w') as f:
        f.write(str(index))
        
current_image_index = load_checkpoint()
print(current_image_index)
if current_image_index != 0:
    current_image_index = 0
    save_checkpoint(current_image_index)
    print("checkpoint is zeroed")