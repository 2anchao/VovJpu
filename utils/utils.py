import random
import os
seed = 6
def split_dataset(imagedir):
    imagenames = os.listdir(imagedir)
    random.seed(6)
    print(imagenames[0:5])
    random.shuffle(imagenames)
    print(imagenames[0:5])


if __name__ == '__main__':
    split_dataset(r'D:\paper\npydata\train\imgs')
