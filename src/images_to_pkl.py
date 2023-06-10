from PIL import Image
import torchvision.transforms as transforms
import torch
import os
import pickle

class DatasetBuilder:
    def __init__(self, pkt_file_name, images_path, ratio) -> None:
        self.pkl_file_name = pkt_file_name
        self.images_path = images_path
        self.ratio = ratio

        self._all_images = {}

        info = f"Dataset builder:\n" +\
               f"Will load {self.ratio}% of the files from {self.images_path}\n" +\
               f"and store them into {self.pkl_file_name}\n"
        
        print(info)

    
    def build(self) -> torch.Tensor:
        transform = transforms.ToTensor()        
        files_in_dir = os.listdir(self.images_path)
        num_files = len(files_in_dir)

        print(f"{self.images_path} contains {num_files} images")
        images_to_load = int(num_files * self.ratio)
        print(f"Loading {images_to_load} images...")
        
        info_every = 100
        for i in range(images_to_load):
            image_file = files_in_dir[i]
            image = Image.open(self.images_path + image_file).convert("RGB")
            tensor_image = transform(image)   
            img_id = int(image_file.split("_")[-1].split(".")[0])
            self._all_images[img_id] = tensor_image
            if i % info_every == 0:
                print(f"Image {i}/{images_to_load}")

    
    def save(self) -> None:
        save_path = "../clean_data/" + self.pkl_file_name
        print(f"Saving dictionary into {save_path}")
        with open(save_path, "wb") as pkl_file:
            pickle.dump(self._all_images, pkl_file)
        print("Saved successfully")
        
coco_year = 2014
images_path = "../data/cocoapi/images/"

data_type = input("Train / validation: ").lower()
while data_type not in ["train", "validation"]:
    data_type = input("Train / validation").lower()
    
file_name = f"dataset_{data_type}{coco_year}.pkl"

ratio = 1

if data_type == "train":
    ratio = 0.1


dataBuilder = DatasetBuilder(file_name, f"{images_path}{data_type}{coco_year}/", ratio)

dataBuilder.build()
dataBuilder.save()


with open("../clean_data/dataset_train2014.pkl", "rb") as pkl:
    dict = pickle.load(pkl)

print(dict)
    


