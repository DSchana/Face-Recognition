import os

os.system("cmake ./Build/")
os.system("make -C ./Build/")
os.system("./Build/FaceRecognition")
