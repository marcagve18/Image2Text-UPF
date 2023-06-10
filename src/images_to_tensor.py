from PIL import Image
import torchvision.transforms as transforms
import torch
import os


class ImagesToTensor:
    def __init__(self, input_folder, output_folder, transforms) -> None:
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.transforms = transforms
        self.images = []

        info = f"Dataset builder:\n" +\
               f"Will load images from {self.input_folder}\n" +\
               f"and store them into {self.output_folder}\n"
        
        print(info)

    
    def build(self) -> torch.Tensor:
         
        files_in_dir = os.listdir(self.input_folder)
        num_files = len(files_in_dir)

        print(f"{self.input_folder} contains {num_files} images")
        images_to_load = num_files
        print(f"Loading {images_to_load} images...")

        info_every = 400
        for i in range(images_to_load):
            image_file = files_in_dir[i]
            image = Image.open(self.input_folder + image_file).convert("RGB")
            tensor_image = self.transforms(image)   
            self.__save(tensor_image= tensor_image, file_name = image_file)
            if i % info_every == 0:
                print(f"Image {i}/{images_to_load}")

    
    def __save(self, tensor_image: torch.Tensor, file_name: str) -> None:
        save_path = self.output_folder + file_name + ".pt"
        torch.save(tensor_image, save_path)
 
        
coco_year = 2017
images_path = "../data/cocoapi/images/"

data_type = input("train / val: ").lower()
while data_type not in ["train", "val"]:
    data_type = input("Train / val").lower()
    
file_name = f"dataset_{data_type}{coco_year}.pt"

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406), 
        (0.229, 0.224, 0.225),
    ),
]) 


dataBuilder = ImagesToTensor(f"{images_path}{data_type}{coco_year}/", f"../clean_data/{data_type}{coco_year}/", transform_train)
dataBuilder.build()

    

