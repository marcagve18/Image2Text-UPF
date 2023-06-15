import matplotlib.pyplot as plt

efficientnet_bidirecional = [[0.09, 0.09, 0.05, 0.05], [0.04, 0.04, 0.03, 0.03], [0.03, 0.03, 0.02, 0.02]]
resnet_loss = [[47.3, 52.88, 52.96, 53.03], [32.61, 34.83, 37.2, 38], [15.35, 16.31, 19.12, 19.22]]
vit = [[31.56, 42.45, 47.64, 50.74], [18.53, 27.34, 30.45, 33.12], [8.49, 11.51, 14.13, 14.99]]
efficientnet = [[16.4, 24.72, 30.74, 31.6], [9.92, 15.8, 18.78, 18.78], [5.75, 7.65, 8.27, 8.17]]

fig, axes = plt.subplots(1, 3, figsize=(18,6))
plt.suptitle("CIDEr evolution", fontsize=14, fontweight="bold")

axes[0].set_title("In-domain")
axes[1].set_title("Near-domain")
axes[2].set_title("Out-domain")
for i in range(3):
    axes[i].plot(efficientnet_bidirecional[i], label="Bidirectional EfficientNetB7")
    axes[i].plot(resnet_loss[i], label="Resnet50")
    axes[i].plot(vit[i], label="ViT")
    axes[i].plot(efficientnet[i], label="EfficientNetB7")
    axes[i].grid()
    axes[i].set_xlabel("Training epochs")
    axes[i].set_ylabel("CIDEr values")
    axes[i].set_xticks([0, 1, 2, 3])
    axes[i].set_xticklabels(['1', '2', '3', '4'])
    axes[i].legend()
    

# plt.legend()
plt.show()