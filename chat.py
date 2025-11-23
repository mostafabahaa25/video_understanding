from frame_extraction import extract_frames
from model import Qwen3VLAgent

agent = Qwen3VLAgent()

video_path = "/kaggle/input/test-2-ego/ego_test3.mp4"
frames = extract_frames(video_path, num_frames=16)

video_captions = []
for i, frame in enumerate(frames):
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
    video_captions.append(output[0])


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

video_summary = agent.chat(summary_prompt)
print(video_summary[0])