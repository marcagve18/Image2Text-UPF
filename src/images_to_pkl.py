from PIL import Image
import torchvision.transforms as transforms
import torch
import os
import pickle

class DatasetBuilder:
    def __init__(self, pkt_file_name, images_path, output_path="../clean_data/") -> None:
        self.pkl_file_name = pkt_file_name
        self.images_path = images_path
        self.output_path = output_path

        self.images = []
        self.images_ids = []

        info = f"Dataset builder:\n" +\
               f"Will load images from {self.images_path}\n" +\
               f"and store them into {self.pkl_file_name}\n"
        
        print(info)

    
    def build(self) -> torch.Tensor:
        transform_train = transforms.Compose([
            # smaller edge of image resized to 256
            transforms.Resize(256),
            # get 224x224 crop from random location
            transforms.RandomCrop(224),
            # horizontally flip image with probability=0.5
            transforms.RandomHorizontalFlip(),
            # convert the PIL Image to a tensor
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),  # normalize image for pre-trained model
                (0.229, 0.224, 0.225),
            ),
        ])  
        files_in_dir = os.listdir(self.images_path)
        num_files = len(files_in_dir)

        print(f"{self.images_path} contains {num_files} images")
        images_to_load = num_files
        print(f"Loading {images_to_load} images...")

        info_every = 400
        for i in range(images_to_load):
            image_file = files_in_dir[i]
            image = Image.open(self.images_path + image_file).convert("RGB")
            tensor_image = transform_train(image)   
            self.__save(tensor_image= tensor_image, file_name = image_file)
            if i % info_every == 0:
                print(f"Image {i}/{images_to_load}")

    
    def __save(self, tensor_image: torch.Tensor, file_name: str) -> None:
        save_path = self.output_path + file_name + ".pt"
        torch.save(tensor_image, save_path)
 
        
coco_year = 2017
images_path = "../data/cocoapi/images/"

data_type = input("Train / validation: ").lower()
while data_type not in ["train", "validation"]:
    data_type = input("Train / validation").lower()
    
file_name = f"dataset_{data_type}{coco_year}.pt"

dataBuilder = DatasetBuilder(file_name, f"{images_path}{data_type}{coco_year}/")
dataBuilder.build()
#dataBuilder.save()

    


