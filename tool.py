import torch
import torchvision
import torch_scatter
import torch_cluster
import torch_sparse

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

# 测试 nms
boxes = torch.rand(10, 4).cuda()
scores = torch.rand(10).cuda()
keep = torchvision.ops.nms(boxes, scores, 0.5)
print("NMS result:", keep)

# 测试 scatter
src = torch.randn(10, 8).cuda()
index = torch.randint(0, 5, (10,), dtype=torch.long).cuda()
out = torch_scatter.scatter(src, index, dim=0, reduce="mean")
print("Scatter output shape:", out.shape)