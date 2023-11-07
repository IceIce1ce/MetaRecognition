import os
import shutil

with open('ucfTrainTestlist/trainlist01.txt', 'r') as f:
    for file in f:
        data = file.split()
        # split train
        output_path = 'train/' + data[0].split('/')[0]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        shutil.copy('UCF-101/' + data[0], output_path)

        # split test
        # output_path = 'validation/' + data[0].split('/')[0]
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)
        # shutil.copy('UCF-101/' + data[0], output_path)