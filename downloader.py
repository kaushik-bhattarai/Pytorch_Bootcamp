import os
from zipfile import ZipFile
from urllib.request import urlretrieve

#downloading asset
def download_and_unzip(url, save_path):
    print(f"downloading and extracting assets....", end="")

    urlretrieve(url, save_path)

    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
        
        print("done")

    except Exception as e:
        print("\nInvalid file.", e)