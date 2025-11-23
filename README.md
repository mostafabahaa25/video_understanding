# video understanding and summarization 

**End-to-End Video Captioning & Summarization Pipeline**

This project provides a complete workflow for processing videos using **Qwen3-VL**, including:

* Downloading videos directly from a URL
* Extracting evenly spaced frames
* Generating per-frame captions using Qwen3-VL
* Producing a concise final summary of the full video

The pipeline is simple, modular, and works on **Kaggle, Colab, and local machines**.

---

## Features

* **Direct video download** from any accessible URL
* **Frame extraction** with OpenCV
* **Qwen3-VL inference** for describing each frame
* **Automatic summary generation** from all frame captions
* Fully **GPU-accelerated** on CUDA devices
* Clean and minimal code: easy to extend into projects or research work

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Required libraries include:

* PyTorch
* Transformers
* Accelerate
* BitsAndBytes
* OpenCV
* Requests
* FAISS / ChromaDB (optional utilities)

---

## Downloading a Video

Use a direct link to download a video into the notebook:

```python
from download import download_video

video_path = download_video("https://example.com/video.mp4")
```

The function automatically handles:

* 403 blocking (user-agent spoofing)
* Invalid URL cleanup
* Large streaming downloads

---

## Extracting Frames

```python
from frame_extraction import extract_frames

frames = extract_frames(video_path, num_frames=16)
```

The extractor returns **RGB frames** ready for Qwen3-VL.

---

## Running Qwen3-VL for Captioning

```python
from model import Qwen3VLAgent

agent = Qwen3VLAgent()

captions = []
for frame in frames:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": "Describe this frame."}
            ],
        }
    ]
    output = agent.chat(messages)
    captions.append(output[0])
```

Each frame receives a natural-language description.

---

## Generating a Final Summary

```python
summary_prompt = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Below are descriptions of several video frames:\n\n"
                    + " ".join(captions)
                    + "\n\nProduce a single, concise summary describing the overall action."
                )
            }
        ]
    }
]

video_summary = agent.chat(summary_prompt)
print(video_summary[0])
```

You receive a **3–4 line summary** describing the whole video.

---

## Project Structure

```
project/
│
├── download.py             # Download video function
├── frame_extraction.py     # Frame extraction helper
├── model.py                # Qwen3-VL agent wrapper
├── chat.py                 # Example pipeline script
├── requirements.txt
└── README.md
```

---

## Notes

* Works best with **Qwen/Qwen3-VL-4B-Instruct**
* On Kaggle, GPU acceleration is automatic when available
* The pipeline can easily be integrated into larger CV or multimodal projects

---

## Future Extensions

* Support YouTube / Google Drive links
* Scene detection for smarter frame sampling
* Full video caption generation (per second / per segment)
* Adding a UI with Gradio or Streamlit
