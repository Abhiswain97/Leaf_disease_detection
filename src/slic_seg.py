from skimage import io
from skimage.segmentation import slic
import os
import glob
import cv2
import numpy as np
import time
import pandas as pd
import joblib
from tqdm import tqdm
import logging
import functools
import matplotlib.pyplot as plt

# logger = logging.getLogger("myapp-10")
# hdlr = logging.FileHandler("logs/myapp-10.log", mode="w")
# formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
# hdlr.setFormatter(formatter)
# logger.addHandler(hdlr)
# logger.setLevel(logging.INFO)


class Superpixels:
    def __init__(self, path):
        self.path = path
        self.masks = []
        self.X, self.y = [], []
        self.mcx, self.mcy = [], []
        self.blb, self.bs, self.ls = [], [], []

    @staticmethod
    def visualize(spxl, mask):
        """
        Visualize a superpixel and mask.

        :param spxl: Image super-pixel
        :param mask: Image super-pixel mask
        """
        cv2.imshow("spxl", spxl)
        cv2.imshow("mask", mask)
        cv2.waitKey(100)

    @staticmethod
    def individual_superpixel(segments, image, mode):
        """
        Separate individual super-pixels.

        :param segments: SLIC segmented image
        :param image: Original image
        :param mode: Specifies whether RGB or Gray-scale

        :return: list of SLIC segments
        """
        superpixels_rgb, superpixels_gs = [], []
        for (i, segVal) in enumerate(np.unique(segments)):
            mask = np.zeros(segments.shape[:2], dtype="uint8")
            mask[segments == segVal] = 255
            segmented_image = cv2.bitwise_and(image, image, mask=mask)

            superpixels_rgb.append(segmented_image)
            superpixels_gs.append(mask)

        if mode == "RGB":
            return superpixels_rgb
        else:
            return superpixels_gs

    def save_pkl(self, segments, name, label, mode="image"):
        """
        :param segments:
        :param name:
        :param label:
        :param mode:
        :return:
        """
        if mode == "image":
            np.save(f"image_superpixel_pkls/{name}_{label}.npy", segments)
            if os.path.exists(f"image_superpixel_pkls/{name}_{label}.pkl"):
                print(f"[Skipping] image_superpixel_pkls/{name}_{label}.pkl")

            else:
                joblib.dump(segments, f"./image_superpixel_pkls/{name}_{label}.pkl")
        else:
            joblib.dump(segments, f"./mask_superpixel_pkls/{name}_{label}.pkl")

    def slic_sp(self):
        """
        Save SLIC images as .pkl files after resizing.
        """
        df_images = pd.read_csv("Image_path_labels.csv")
        df_masks = pd.read_csv("Mask_path_labels.csv")

        image_paths = df_images["paths"].values
        mask_paths = df_masks["paths"].values

        if os.path.exists("image_superpixel_pkls"):
            print("[Directory exits] saving pkl files....")
        else:
            os.mkdir("./image_superpixel_pkls")

        for image_path in image_paths:
            image = io.imread(image_path)
            image = np.resize(image, (256, 256, 3))
            segments = slic(image, 100)
            self.save_pkl(
                segments,
                image_path.split("\\")[-1].split(".")[0],
                image_path.split("\\")[-2],
            )

    def make_label(self, image_paths, mask_paths, image_spxl_paths, mode="gray"):
        """
        Make binary(Diseased/Non-Diseased) classification labels &
        multi-class classification labels.

        This functions loads the segmented SLIC super-pixels,
        then it separates them individually. After that it loops through every segment,
        then labels then for binary classification (0 or 1).
        For multi-class classification (0, 1, 2, 3), we check if a super-pixel has any pixel intensity > 0,
        if yes then the super-pixel is infected label as 0 & based on super-pixel name we label them (0, 1, 2, 3)

        :param image_paths: list of image paths
        :param mask_paths: list of mask paths
        :param image_spxl_paths: list of SLIC image paths
        :param mode: RGB or gray super-pixels
        """
        for spxl_path in tqdm(image_spxl_paths, total=len(image_spxl_paths)):
            spxlname = spxl_path.split("\\")[-1][
                : spxl_path.split("\\")[-1].rindex("_")
            ]
            for image_path, mask_path in zip(image_paths, mask_paths):

                imagename = image_path.split("\\")[-1].split(".")[0]
                maskname = mask_path.split("\\")[-1].split(".")[0]

                if spxlname == imagename and spxlname == maskname:

                    logger.info(
                        f"{spxlname}(superpixel name) matches {imagename}(image) and {maskname}(mask)"
                    )
                    folder = spxl_path.split("_")[-1].split(".")[0]

                    logger.info("[Loading images]")
                    image = cv2.imread(image_path)
                    mask = cv2.imread(mask_path, 0)
                    slic_image = joblib.load(os.path.join(spxl_path))

                    logger.info("[Separating segments....]")
                    if mode == "RGB":
                        superpixels = self.individual_superpixel(
                            slic_image, image, mode
                        )
                    else:
                        superpixels = self.individual_superpixel(
                            slic_image, image, mode
                        )

                    for i, spxl in tqdm(enumerate(superpixels), total=len(superpixels)):

                        # np.save(f'Labels\\RGB_superpixels\\{spxlname}_{folder}_{i+1}_binary.npy', spxl)
                        # hkl.dump(spxl, f'Labels\\hickle_labels\\{spxlname}_{folder}_{i + 1}_binary.hkl')

                        res = spxl * mask

                        if len(np.unique(res)) >= 2:

                            # np.save(f'Labels\\RGB_superpixels\\{spxlname}_{folder}_{i + 1}_multiclass.npy', spxl)
                            # np.savetxt(f'Labels\\mask_ground_truth\\{spxlname}_{folder}_{i + 1}_gt.txt', res)
                            logger.info(
                                f"Labels\\mask_ground_truth\\{spxlname}_{folder}_{i + 1}_gt.npy [SAVED]"
                            )
                            # hkl.dump(spxl, f'Labels\\hickle_labels\\{spxlname}_{folder}_{i + 1}_multiclass.hkl')

                            self.y.append(1)

                            if folder == "Bacterial leaf blight":
                                self.mcy.append(0)
                                logger.info(
                                    f"Superpixel {i + 1} is diseased with Bacterial Leaf blight"
                                )

                            if folder == "Brown spot":
                                self.mcy.append(1)
                                logger.info(
                                    f"Superpixel {i + 1} is diseased with Brown spot"
                                )

                            if folder == "Leaf smut":
                                self.mcy.append(2)
                                logger.info(
                                    f"Superpixel {i + 1} is diseased with Leaf smut"
                                )

                        else:
                            self.y.append(0)

                    logger.info(f"{spxlname} done \n")

        logger.info("[SAVING hickle files] .....")

        # np.save(f'Labels\\RGB_superpixels\\binary_labels(RGB).npy', self.y)
        # hkl.dump(f'Labels\\hickle_labels\\binary_labels(RGB).hkl', self.y)

        logger.info("[Binary classification superpixels and labels have been saved]")

        # np.save(f'Labels\\RGB_superpixels\\multiclass_labels(RGB).npy', self.mcy)
        # hkl.dump(f'Labels\\hickle_labels\\binary_labels(RGB).hkl', self.mcy)

        logger.info(
            "[Multiclass classification superpixels and labels have been saved]"
        )

    def set_images_and_masks(self):
        """

        """
        start = time.time()
        try:
            df_masks = pd.read_csv("Mask_path_labels.csv")
            df_images = pd.read_csv("Image_path_labels.csv")

            mask_paths = df_masks["paths"].values
            image_paths = []
            image_spxl_paths = []
            for image_path in df_images["paths"].values:
                image_name = image_path.split("\\")[-1].split(".")[0]
                for mask_path in df_masks["paths"].values:
                    mask_name = mask_path.split("\\")[-1].split(".")[0]
                    if image_name == mask_name:
                        image_paths.append(image_path)

            for file in os.listdir("image_superpixel_pkls"):
                for mask_path in df_masks["paths"].values:
                    mask_name = mask_path.split("\\")[-1].split(".")[0]
                    if file[: file.rindex("_")] == mask_name:
                        image_spxl_paths.append(
                            os.path.join(
                                r"C:\Users\Abhishek Swain\PycharmProjects\leaf-disease\image_superpixel_pkls",
                                file,
                            )
                        )

            print(len(image_paths), len(mask_paths), len(image_spxl_paths))

            self.make_label(image_paths, mask_paths, image_spxl_paths)

        except Exception as e:
            logger.error("Exception at %s", exc_info=e)

        logger.info(f"[COMPLETED] in {(time.time() - start) / 60} minutes")

    def make_csv(self, path, org_folders, mask_folders, mode="org"):
        labels = {}
        paths1, paths2 = [], []
        label1, label2 = [], []

        for folder in os.listdir(path):
            if folder in mask_folders:
                for file in os.listdir(os.path.join(path, folder)):
                    if file.endswith(".pgm"):
                        paths1.append(os.path.join(path, folder + "\\" + file))
                        label1.append(folder[: folder.rindex(" ") + 1])

            if folder in org_folders:
                for file in os.listdir(os.path.join(path, folder)):
                    print(file)
                    paths2.append(os.path.join(path, folder + "\\" + file))
                    label2.append(folder)

        if mode == "mask":
            labels["paths"] = paths1
            labels["label"] = label1
            df = pd.DataFrame.from_dict(labels)
            df.to_csv("Massk_path_labels.cv")

        if mode == "org":
            labels["paths"] = paths2
            labels["label"] = label2
            df = pd.DataFrame.from_dict(labels)
            df.to_csv("Image_path_labels.csv")


if __name__ == "__main__":
    org_folders = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]
    mask_folders = [
        "Bacterial Leaf blight masks",
        "Brown spot masks",
        "leaf smut masks",
    ]
    path = "Leaf_disease"
    sp = Superpixels(path)
    sp.make_csv(path, org_folders, mask_folders, mode="org")
    # sp.make_image_superpixel()
    # sp.slic_sp()
    # sp.binary_class()
    # sp.binary_label()
    # sp.set_images_and_masks()
