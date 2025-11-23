import requests

def download_video(url, save_path="kaggle/working/downloaded_video.mp4"):
    print(f"Downloading video from: {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Video saved to: {save_path}")
    return save_path
    
def clean_url(url):
    return url.strip().strip('"').strip("'")
