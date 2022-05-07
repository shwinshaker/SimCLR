import torch
from vit_pytorch.vit_for_small_dataset import ViT

__all__ = ['vit']

def vit(depth, width=1, num_classes=10, dropRate=0.0, heads=6, patch_size=16):
    return ViT(image_size=32, # 256,
                patch_size=patch_size,
                num_classes=num_classes,
                dim=int(256 * width), # 1024,
                depth=depth,
                heads=heads, # 12,
                mlp_dim=int(512 * width), # 2048,
                dropout=dropRate,
                emb_dropout=dropRate)
    

def test():
    net = vit(depth=3, width=1, num_classes=10).cuda()
    print("     Total params: %.2fM" % (sum(p.numel() for p in net.parameters())/1000000.0))
    scaler = sum(p.numel() for p in net.parameters())

    y = net(torch.randn(64, 3, 32, 32).cuda())
    print(y.size())


if __name__ == '__main__':
    test()
