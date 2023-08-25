import numpy as np
import os

# For checking out that how many images are available in the train set we can use import OS
for types in os.listdir("/content/gdrive/MyDrive/Pro coders (7,8)/PC_Session-66/PlantDisease/train_set"):
    print(str(len(os.listdir("/content/gdrive/MyDrive/Pro coders (7,8)/PC_Session-66/PlantDisease/train_set/"+ types)))+" "+ types+' images')
