import os
from typing import Optional

import moviepy.editor as mp
from pytube import YouTube


def download_youtube_video(url: str, output_path: Optional[str] = None) -> None:
    """
    Downloads a YouTube video from the given URL and saves it to the specified output path or the current directory.

    Args:
        url: The URL of the YouTube video to download.
        output_path: The path where the downloaded video will be saved. If None, the video will be saved to the current
        directory.

    Returns:
        None
    """
    yt = YouTube(url)

    video_stream = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )

    if output_path:
        video_stream.download(output_path)
        print(f"Video successfully downloaded to {output_path}")
    else:
        video_stream.download()
        print("Video successfully downloaded to the current directory")


def convert_to_wav(input_file: str, output_file: Optional[str] = None) -> None:
    """
    Converts an audio file to WAV format using FFmpeg.
    Args:
        input_file (str): The path of the input audio file to convert.
        output_file (str): The path of the output WAV file. If None, the output file will be created by replacing the input file
        extension with ".wav".
    Returns:
        None
    """
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + ".wav"

    clip = mp.VideoFileClip(input_file)
    clip.audio.write_audiofile(output_file, codec="pcm_s16le")
