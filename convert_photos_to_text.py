import os
from PIL import Image
import numpy

num_of_images = len(os.listdir(f"./char_recognition/char_pictures"))
file = open("./char_recognition/char_complete.txt","w")
for i in range(num_of_images):
        image = Image.open(f"./char_recognition/char_pictures/{i}.png")
        image_array = numpy.array(image)
        binary_array = numpy.where(image_array > 127, 1, -1).flatten()
        counter = 1
        for data in binary_array:
                if(counter == 120):
                        file.write(f"{data}\n")
                else:
                        file.write(f"{data},")
                        counter +=1
file.close()