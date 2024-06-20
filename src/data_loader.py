import os
import urllib.request

def download_data(urls, dest_folder=r"data/"):
    os.makedirs(dest_folder, exist_ok=True)
    for url in urls:
        file_name = url.split("/")[-1]
        urllib.request.urlretrieve(url, os.path.join(dest_folder, file_name))
    print("Files downloaded and organized successfully.")

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.lower().strip().split(' ||| ')
            data.append(line)
    return data