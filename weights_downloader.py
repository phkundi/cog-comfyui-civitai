import subprocess
import time
import os
from weights_manifest import WeightsManifest
from config import config
import requests
from tqdm import tqdm

MODELS_PATH = config["MODELS_PATH"]

class WeightsDownloader:
    supported_filetypes = [
        ".ckpt",
        ".safetensors",
        ".pt",
        ".pth",
        ".bin",
        ".onnx",
        ".torchscript",
        ".engine",
        ".patch"
    ]

    def __init__(self):
        self.weights_manifest = WeightsManifest()
        self.weights_map = self.weights_manifest.weights_map

    def get_weights_by_type(self, type):
        return self.weights_manifest.get_weights_by_type(type)

    def download_weights(self, weight_str):
        if weight_str in self.weights_map:
            if self.weights_manifest.is_non_commercial_only(weight_str):
                print(
                    f"⚠️  {weight_str} is for non-commercial use only. Unless you have obtained a commercial license.\nDetails: https://github.com/fofr/cog-comfyui/blob/main/weights_licenses.md"
                )

            if isinstance(self.weights_map[weight_str], list):
                for weight in self.weights_map[weight_str]:
                    self.download_if_not_exists(
                        weight_str, weight["url"], weight["dest"]
                    )
            else:
                self.download_if_not_exists(
                    weight_str,
                    self.weights_map[weight_str]["url"],
                    self.weights_map[weight_str]["dest"],
                )
        # else:
        #     raise ValueError(
        #         f"{weight_str} unavailable. View the list of available weights: https://github.com/fofr/cog-comfyui/blob/main/supported_weights.md"
        #     )

    def check_if_file_exists(self, weight_str, dest):
        if dest.endswith(weight_str):
            path_string = dest
        else:
            path_string = os.path.join(dest, weight_str)
        return os.path.exists(path_string)

    def download_if_not_exists(self, weight_str, url, dest):
        if self.check_if_file_exists(weight_str, dest):
            print(f"✅ {weight_str} exists in {dest}")
            return
        WeightsDownloader.download(weight_str, url, dest)
    

    @staticmethod
    def download(weight_str, url, dest):
        if "/" in weight_str:
            subfolder = weight_str.rsplit("/", 1)[0]
            dest = os.path.join(dest, subfolder)
            os.makedirs(dest, exist_ok=True)

        print(f"⏳ Downloading {weight_str} to {dest}")
        start = time.time()
        subprocess.check_call(
            ["pget", "--log-level", "warn", "-xf", url, dest], close_fds=False
        )
        elapsed_time = time.time() - start
        try:
            file_size_bytes = os.path.getsize(
                os.path.join(dest, os.path.basename(weight_str))
            )
            file_size_megabytes = file_size_bytes / (1024 * 1024)
            print(
                f"✅ {weight_str} downloaded to {dest} in {elapsed_time:.2f}s, size: {file_size_megabytes:.2f}MB"
            )
        except FileNotFoundError:
            print(f"✅ {weight_str} downloaded to {dest} in {elapsed_time:.2f}s")
        
    def handle_civitai_download(self, model):
        url = model['url'] + f"&token={config['CIVITAI_API_TOKEN']}"
        dest_folder = os.path.join(MODELS_PATH, model['dest'])
        os.makedirs(dest_folder, exist_ok=True)

        head_response = requests.head(url, allow_redirects=True)
        if 'Content-Disposition' in head_response.headers:
            content_disposition = head_response.headers['Content-Disposition']
            filename = content_disposition.split('filename=')[1].strip('"')
        else:
            # Fallback to the name provided in weight_info if header is not available
            filename = model['name']
        
        dest_path = os.path.join(dest_folder, filename)

        # check if exists locally
        if os.path.exists(dest_path):
            print(f"✅ {filename} exists in {dest_folder}")
            return
        
        # if not, download
        print(f"⏳ Downloading {filename} to {dest_folder}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8 KB

        with open(dest_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                progress_bar.update(size)

        print(f"✅ {filename} downloaded to {dest_folder}")