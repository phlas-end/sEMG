import socket
import numpy as np

# ===== 配置 =====
SERVER_IP = "192.168.3.9"   # 修改成 ESP32 的实际 IP
SERVER_PORT = 3333

# ===== 加载数据 =====
X = np.load(r"D:\Project\sEMG\test_data\E2_X.npy").astype(np.float32)   # (n,16,200)
y = np.load(r"D:\Project\sEMG\test_data\E2_y.npy").astype(np.int32)     # (n,)

n = X.shape[0]
print(f"Loaded dataset: {n} samples")

# ===== 统计正确率 =====
correct = 0

for i in range(n):
    # 取第 i 个样本
    sample = X[i]  # (16,200)
    label = int(y[i])

    # 建立连接
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))

    # 发送数据
    sock.sendall(sample.tobytes())

    # 接收预测结果 (int, 类别索引)
    recv_bytes = sock.recv(4)   # int32 = 4字节
    pred_class = int.from_bytes(recv_bytes, byteorder="little", signed=True)

    sock.close()

    # 比对
    if pred_class == label:
        correct += 1

    # 可选：打印进度
    if (i+1) % 50 == 0 or i == n-1:
        acc = correct / (i+1)
        print(f"Processed {i+1}/{n}, current accuracy = {acc:.4f}")

# ===== 最终准确率 =====
accuracy = correct / n
print(f"\nFinal accuracy: {accuracy:.4f} ({correct}/{n})")
