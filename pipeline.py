import os
from model import Qwen3VLAgent
from frame_extraction import extract_frames
from video_downloader import download_video, clean_url


def run_video_summary(video_url: str, num_frames: int = 16) -> str:
    # 1 — Clean and download the video
    # Check if the input is a local file
    if os.path.exists(video_url):
        video_path = video_url  # use local file directly
    else:
        video_url = clean_url(video_url)
        video_path = download_video(video_url)
    # 2 — Load the vision-language model
    agent = Qwen3VLAgent()

    # 3 — Extract frames
    frames = extract_frames(video_path, num_frames=num_frames)

    # 4 — Caption each frame
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

    # 5 — Summarize all captions
    summary_prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Below are descriptions of several video frames:\n\n"
                        + " ".join(video_captions)
                        + "\n\n"
                        "Your task: Produce a single, concise summary that captures the main idea. "
                        "Avoid repeating details,and avoid describing each description separately. "
                        "Combine everything into **one short paragraph** that represents the overall activity or situation. "
                        "Do NOT exceed 3–4 lines."
                    )
                }
            ]
        }
    ]
    summary = agent.chat(summary_prompt)[0]
    return summary
