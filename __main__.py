from pipeline import run_video_summary

def main(video_url=None):
    if video_url is None:
        video_url = input("Paste the video link: ").strip()
    result = run_video_summary(video_url)
    print("\n=== VIDEO SUMMARY ===\n")
    print(result)
    
if __name__ == "__main__":
    main()
