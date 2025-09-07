import os

folder_path = '/data/inbox'  

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

if files:
    print("Нашёл файлы:", files)
else:
    print("Файлов нет")