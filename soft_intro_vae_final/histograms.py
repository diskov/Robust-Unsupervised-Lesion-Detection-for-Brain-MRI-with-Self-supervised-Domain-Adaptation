import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sb
from IPython import embed


def plot(x, name):
    first_half = x.shape[0]//2
    colors = np.ones([x.shape[0]//2])
    colors_camcan = np.zeros([x.shape[0]//2])
    colors_final = np.hstack([colors_camcan, colors])
    palette = np.array(sb.color_palette("hls", 10))  # Choosing color palette

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:first_half, 0], x[:first_half, 1], lw=0, s=40, c=palette[colors_camcan.astype(np.int)])
    sc = ax.scatter(x[first_half+1:, 0], x[first_half+1:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    ax.legend(["camcan", "brats_t2"])
    # Add the labels for each digit.
    txts = []
    # for i in range(2):
    #     # Position of each label.
    #     xtext, ytext = np.median(x[colors_final == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
    #     txts.append(txt)
    plt.savefig(f'./TSNE/Figures/{name}.png')
    return f, ax


def plot3d(x, name):
    first_half = x.shape[0]//2
    colors = np.ones([x.shape[0]//2])
    colors_camcan = np.zeros([x.shape[0]//2])
    colors_final = np.hstack([colors_camcan, colors])
    palette = np.array(sb.color_palette("hls", 10))  # Choosing color palette

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax = plt.axes(projection="3d")
    sc = ax.scatter3D(x[:first_half, 0], x[:first_half, 1], x[:first_half, 2], lw=0, s=40, c=palette[colors_camcan.astype(np.int)])
    sc = ax.scatter3D(x[first_half+1:, 0], x[first_half+1:, 1], x[first_half+1:, 2], lw=0, s=40, c=palette[colors.astype(np.int)])
    ax.legend(["camcan", "brats_t2"])

    # Add the labels for each digit.
    txts = []
    # for i in range(10):
    #     # Position of each label.
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
    #     txts.append(txt)
    # plt.show()
    plt.savefig(f'./TSNE/Figures/{name}.png')
    return f, ax


# t2_data_tsne = np.loadtxt(f"./X_numpy_tsne_1st_n_iter_5k.txt")
# t2_data_tsne_da = np.loadtxt(f"./TSNE/before/t2_X_numpy_tsne_perp_50_iter_50k_10_batches_PCA_compon_2.txt")
# t2_data_tsne_da_100 = np.loadtxt(f"./TSNE/before/t2_X_numpy_tsne_perp_100_iter_50k_10_batches_PCA_compon_2.txt")
# t2_data_tsne_da_200 = np.loadtxt(f"./TSNE/before/t2_X_numpy_tsne_perp_200_iter_50k_10_batches_PCA_compon_2.txt")
# t2_data_tsne_da_250 = np.loadtxt(f"./TSNE/before/t2_X_numpy_tsne_perp_250_iter_50k_10_batches_PCA_compon_2.txt")
t2_data_tsne_da_150 = np.loadtxt(f"./TSNE/before/t1_X_numpy_tsne_perp_150_iter_1k_all_batches_PCA_compon_2.txt")
# camcan_data_tsne = np.loadtxt(f"./camcan_X_numpy_tsne_n_iter_5k.txt")
# camcan_data_tsne_da = np.loadtxt(f"./TSNE/before/camcan_X_numpy_tsne_perp_50_iter_50k_10_batches_PCA_compon_2.txt")
# camcan_data_tsne_da_100 = np.loadtxt(f"./TSNE/before/camcan_X_numpy_tsne_perp_100_iter_50k_10_batches_PCA_compon_2.txt")
# camcan_data_tsne_da_200 = np.loadtxt(f"./TSNE/before/camcan_X_numpy_tsne_perp_200_iter_50k_10_batches_PCA_compon_2.txt")
# camcan_data_tsne_da_250 = np.loadtxt(f"./TSNE/before/camcan_X_numpy_tsne_perp_250_iter_50k_10_batches_PCA_compon_2.txt")
camcan_data_tsne_da_150 = np.loadtxt(f"./TSNE/before/camcan_X_numpy_tsne_perp_150_iter_1k_all_batches_PCA_compon_2.txt")
# data1 = np.loadtxt(r"./10_3_20220/likelihoods/3000/elbo_data_camcan_model_epoch_51_iter_30345.txt")
# data2 = np.loadtxt(r"./10_3_20220/likelihoods/3000/elbo_data_camcan_gauss_model_epoch_51_iter_30345.txt")
# data3 = np.loadtxt(r"./10_3_20220/likelihoods/3000/elbo_data_mnist_test_model_epoch_51_iter_30345.txt")

# data_t2hm_healthy = np.loadtxt(r"./10_3_20220/likelihoods/3000/elbo_data_t2_hm_healthy_model_epoch_51_iter_30345.txt")
# data_t2hm_lesion = np.loadtxt(r"./10_3_20220/likelihoods/3000/elbo_data_t2_hm_lesion_model_epoch_51_iter_30345.txt")
#
# data_t2_healthy = np.loadtxt(r"./10_3_20220/likelihoods/3000/elbo_data_t2_healthy_model_epoch_51_iter_30345.txt")
# data_t2_lesion = np.loadtxt(r"./10_3_20220/likelihoods/3000/elbo_data_t2_lesion_model_epoch_51_iter_30345.txt")
#
# data_t1hm_healthy = np.loadtxt(r"./10_3_20220/likelihoods/3000/elbo_data_t1_hm_healthy_model_epoch_51_iter_30345.txt")
# data_t1hm_lesion = np.loadtxt(r"./10_3_20220/likelihoods/3000/elbo_data_t1_hm_lesion_model_epoch_51_iter_30345.txt")
#
# data_t1_healthy = np.loadtxt(r"./10_3_20220/likelihoods/3000/elbo_data_t1_healthy_model_epoch_51_iter_30345.txt")
# data_t1_lesion = np.loadtxt(r"./10_3_20220/likelihoods/3000/elbo_data_t1_lesion_model_epoch_51_iter_30345.txt")
# # print(sum(np.isinf(data1)))
# # print(sum(np.isinf(data2)))
# # print(sum(np.isinf(data3)))
# #
# # # data1 = data1[np.any([1000 > data1 > -2000])]
# data_t2hm_healthy = data_t2hm_healthy[data_t2hm_healthy < 500]
# # # data1 = data1[data1 > -15000]
# data_t2hm_lesion = data_t2hm_lesion[data_t2hm_lesion < 500]
# # # data2 = data2[data2 > -15000]
# data_t2_healthy = data_t2_healthy[data_t2_healthy < 500]
# data_t2_lesion = data_t2_lesion[data_t2_lesion < 500]
# data_t1hm_healthy = data_t1hm_healthy[data_t1hm_healthy < 500]
# data_t1hm_lesion = data_t1hm_lesion[data_t1hm_lesion < 500]
# data_t1_healthy = data_t1_healthy[data_t1_healthy < 500]
# data_t1_lesion = data_t1_lesion[data_t1_lesion < 500]
# # # data3 = data3[data3 > -15000]
# #
# data1.tolist()
# # data2.tolist()
# # data3.tolist()
# data_t2hm_healthy.tolist()
# data_t2hm_lesion.tolist()
# data_t2_healthy.tolist()
# data_t2_lesion.tolist()
# data_t1hm_healthy.tolist()
# data_t1hm_lesion.tolist()
# data_t1_healthy.tolist()
# data_t1_lesion.tolist()

# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)
# _ = ax.hist(data1, bins=20, density=True, ec='black', alpha=0.4, color='gray', label='CAMCAN-VAL')
# _ = ax.hist(data2, bins=20, density=True, ec='black', alpha=0.4, label='CAMCAN-LESIONAL')
# _ = ax.hist(data3, bins=20, density=True, ec='black', alpha=0.4, color='red', label='MNIST-TEST')
# # _ = ax.hist(data_t2_healthy, bins=20, density=True, ec='black', alpha=0.4, color='blue', label='BRATS-T2-HEALTHY')
# # _ = ax.hist(data_t2_lesion, bins=20, density=True, ec='black', alpha=0.4, color='green', label='BRATS-T2-LESIONAL')
# ax.grid()
# # ax.set_xlabel('log p(x)')
# ax.set_xlabel('ELBO(x)')
# ax.legend()
# plt.savefig(r'./10_3_20220/likelihoods/3000/elbo_histogram_camcan_ood_datasets.png')
#
# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)
# _ = ax.hist(data1, bins=20, density=True, ec='black', alpha=0.4, color='gray', label='CAMCAN-VAL')
# _ = ax.hist(data2, bins=20, density=True, ec='black', alpha=0.4, label='CAMCAN-LESIONAL')
# # _ = ax.hist(data3, bins=20, density=True, ec='black', alpha=0.4, color='red', label='MNIST-TEST')
# # _ = ax.hist(data_t2_healthy, bins=20, density=True, ec='black', alpha=0.4, color='blue', label='BRATS-T2-HEALTHY')
# # _ = ax.hist(data_t2_lesion, bins=20, density=True, ec='black', alpha=0.4, color='green', label='BRATS-T2-LESIONAL')
# ax.grid()
# # ax.set_xlabel('log p(x)')
# ax.set_xlabel('ELBO(x)')
# ax.legend()
# plt.savefig(r'./10_3_20220/likelihoods/3000/elbo_histogram_camcan_cam_lesional.png')
#
# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)
# _ = ax.hist(data1, bins=20, density=True, ec='black', alpha=0.4, color='gray', label='CAMCAN-VAL')
# # _ = ax.hist(data2, bins=20, density=True, ec='black', alpha=0.6, label='CAMCAN-LESIONAL')
# # _ = ax.hist(data3, bins=20, density=True, ec='black', alpha=0.6, color='red', label='MNIST-TEST')
# _ = ax.hist(data_t2_healthy, bins=20, density=True, ec='black', alpha=0.4, color='blue', label='BRATS-T2-HEALTHY')
# _ = ax.hist(data_t2_lesion, bins=20, density=True, ec='black', alpha=0.4, color='green', label='BRATS-T2-LESIONAL')
# ax.grid()
# # ax.set_xlabel('log p(x)')
# ax.set_xlim([0, 500])
# ax.set_xlabel('ELBO(x)')
# ax.legend()
# plt.savefig(r'./10_3_20220/likelihoods/3000/elbo_histogram_camcan_ood_datasets_t2.png')
# # plt.show()
#
# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)
# _ = ax.hist(data1, bins=20, density=True, ec='black', alpha=0.4, color='gray', label='CAMCAN-VAL')
# # _ = ax.hist(data2, bins=20, density=True, ec='black', alpha=0.6, label='CAMCAN-LESIONAL')
# # _ = ax.hist(data3, bins=20, density=True, ec='black', alpha=0.6, color='red', label='MNIST-TEST')
# _ = ax.hist(data_t2hm_healthy, bins=20, density=True, ec='black', alpha=0.4, color='blue', label='BRATS-T2HM-HEALTHY')
# _ = ax.hist(data_t2hm_lesion, bins=20, density=True, ec='black', alpha=0.4, color='green', label='BRATS-T2HM-LESIONAL')
# ax.grid()
# # ax.set_xlabel('log p(x)')
# ax.set_xlim([0, 500])
# ax.set_xlabel('ELBO(x)')
# ax.legend()
# plt.savefig(r'./10_3_20220/likelihoods/3000/elbo_histogram_camcan_ood_datasets_t2hm.png')
#
# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)
# _ = ax.hist(data1, bins=20, density=True, ec='black', alpha=0.4, color='gray', label='CAMCAN-VAL')
# # _ = ax.hist(data2, bins=20, density=True, ec='black', alpha=0.6, label='CAMCAN-LESIONAL')
# # _ = ax.hist(data3, bins=20, density=True, ec='black', alpha=0.6, color='red', label='MNIST-TEST')
# _ = ax.hist(data_t1hm_healthy, bins=20, density=True, ec='black', alpha=0.4, color='blue', label='BRATS-T1HM-HEALTHY')
# _ = ax.hist(data_t1hm_lesion, bins=20, density=True, ec='black', alpha=0.4, color='green', label='BRATS-T1HM-LESIONAL')
# ax.grid()
# # ax.set_xlabel('log p(x)')
# ax.set_xlim([0, 500])
# ax.set_xlabel('ELBO(x)')
# ax.legend()
# plt.savefig(r'./10_3_20220/likelihoods/3000/elbo_histogram_camcan_ood_datasets_t1hm.png')
#
# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)
# _ = ax.hist(data1, bins=20, density=True, ec='black', alpha=0.4, color='gray', label='CAMCAN-VAL')
# # _ = ax.hist(data2, bins=20, density=True, ec='black', alpha=0.6, label='CAMCAN-LESIONAL')
# # _ = ax.hist(data3, bins=20, density=True, ec='black', alpha=0.6, color='red', label='MNIST-TEST')
# _ = ax.hist(data_t1_healthy, bins=20, density=True, ec='black', alpha=0.4, color='blue', label='BRATS-T1-HEALTHY')
# _ = ax.hist(data_t1_lesion, bins=20, density=True, ec='black', alpha=0.4, color='green', label='BRATS-T1-LESIONAL')
# ax.grid()
# # ax.set_xlabel('log p(x)')
# ax.set_xlim([0, 500])
# ax.set_xlabel('ELBO(x)')
# ax.legend()
# plt.savefig(r'./10_3_20220/likelihoods/3000/elbo_histogram_camcan_ood_datasets_t1.png')

# fig = plt.figure()
# # plt.xlim(-3400, -2800)
# plt.hist(data1, weights=np.ones(len(data1)) / len(data1), bins=100, alpha=0.5, label="camcan")
# plt.hist(data2, weights=np.ones(len(data2)) / len(data2), bins=100, alpha=0.5, label="camcan_val")
# plt.hist(data3, weights=np.ones(len(data3)) / len(data1), bins=100, alpha=0.5, label="mnist_test")
# plt.legend()
# plt.savefig(r'./camcan_histograms/my_histogram_paper_dataset_epoch10.png')
# plt.show()
# t2_data_tsne_before_150 = np.loadtxt(f"./TSNE/before/t2_X_numpy_tsne_perp_150_iter_1k_all_batches_PCA_compon_2.txt")
# camcan_data_tsne_before_150 = np.loadtxt(f"./TSNE/before/camcan_X_numpy_tsne_perp_150_iter_1k_all_batches_PCA_compon_2.txt")

if __name__ == "__main__":
    # plot(np.vstack([camcan_data_tsne, t2_data_tsne]), "t2_latent_space_tsne_before_da")
    # plot(np.vstack([camcan_data_tsne_da, t2_data_tsne_da]), "t2_latent_space_tsne_before_da_perp_50_50k_10_batches_PCA_compon_2")
    # plot(np.vstack([camcan_data_tsne_da_100, t2_data_tsne_da_100]), "t2_latent_space_tsne_before_da_perp_100_50k_10_batches_PCA_compon_2")
    plot(np.vstack([camcan_data_tsne_da_150, t2_data_tsne_da_150]), "t1_latent_space_tsne_before_da_perp_150_1k_all_batches_PCA_compon_2")
    # plot(np.vstack([camcan_data_tsne_before_150, t2_data_tsne_before_150]), "t2_latent_space_tsne_before_da_perp_150_1k_all_batches_PCA_compon_2")
    # plot(np.vstack([camcan_data_tsne_da_200, t2_data_tsne_da_200]), "t2_latent_space_tsne_before_da_perp_200_50k_10_batches_PCA_compon_2")
    # plot(np.vstack([camcan_data_tsne_da_250, t2_data_tsne_da_250]), "t2_latent_space_tsne_before_da_perp_250_50k_10_batches_PCA_compon_2")
