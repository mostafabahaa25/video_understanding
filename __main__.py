from pipeline import run_video_summary

def main(video_url=None, num_frames=16):
    if video_url is None:
        video_url = input("Paste the video link: ").strip()
    result = run_video_summary(video_url, num_frames=num_frames)
    print("\n=== VIDEO SUMMARY ===\n")
    print(result)

    
# if __name__ == "__main__":
#     main()
