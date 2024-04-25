import base64
import cv2
import logging
import openai
import requests
import time
from typing import Optional

from IPython.display import display, Image
from moviepy.editor import VideoFileClip, AudioFileClip

logging.basicConfig(level=logging.INFO)


def convert_frames_to_base64(path_to_video: str, resize_dim: Optional[tuple] = None) -> list:
    """
    Reads a video from the path provided and convert it to base64
    If resize_dim is provided then it also resizes the frames
    Args:
        path_to_video (str): video location
        resize_dim (tuple, optional): Resize dimension. Defaults to None.
    Returns:
        list: returns base64 frames in a list
    """
    video = cv2.VideoCapture(path_to_video)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        
        # check if there is any frame to append
        if not success:
            break
        
        # resize the frame in case a dimension is provided
        if resize_dim is not None:
            frame = cv2.resize(frame, resize_dim)
        
        # encode frame to base64
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    logging.getLogger().info(f"{len(base64Frames)}, frames read.")

    return base64Frames


def render_video(frames_list: list) -> None:
    """
    Receives a list of frames and renders them in the notebook cell
    Args:
        frames_list (list): base64 frames
    """
    
    display_handle = display(None, display_id=True)
    for img in frames_list:
        display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
        time.sleep(0.025)


def attach_audio_to_video(audio_path: str, video_path: str, save_path: str) -> None:
    """
    Attaches the audio created to the video
    Args:
        audio_path (str): audio location
        video_path (str): video location
        save_path (str): path to the new video
    """

    # Open the video and audio
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    # Concatenate the video clip with the audio clip
    final_clip = video_clip.set_audio(audio_clip)

    # save final video
    final_clip.write_videofile(save_path)


def get_description(prompt: str, frame_list: list, frame_interval: int, open_ai_key: str, max_tokens: int) -> str:
    """
    Make a call to OpenAI API to get a description from GPT-4V(ision) based on text and images
    Args:
        prompt (str): string with instructions for GPT-4
        frame_list (list): frames to input in GPT-4
        frame_interval (int): the interval between frames we want to pass, it is useful to reduce the number of tokens passed
        open_ai_key (str): you OpenAI key
        max_tokens (int): maximum tokens from GPT-4 response
    Returns:
        str: output from GPT-4
    """

    prompt_message = [
        {
            "role": "user",
            "content": [prompt,
                *map(lambda x: {"image": x}, frame_list[0::frame_interval]),
            ],
        },
    ]

    params = {
        "model": "gpt-4-vision-preview",
        "messages": prompt_message,
        "api_key": open_ai_key,
        "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": max_tokens,
    }

    result = openai.ChatCompletion.create(**params)
    logging.getLogger().info(result.choices[0].message.content)
    return result.choices[0].message.content


def transform_text_to_speech(description: str, open_ai_key: str, model: str, voice: str, save_path: str) -> bytes:
    """
    Receives a string and transform it into audio
    Args:
        description (str): output from GPT-4
        open_ai_key (str): your open ai key
        model (str): TTS model
        voice (str): voice from TTS OpenAI 
        save_path (str): path to save audio
    """

    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {open_ai_key}",
        },
        json={
            "model": model,
            "input": description,
            "voice": voice,
        },
    )

    audio = b""
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio += chunk

    with open(save_path, 'wb') as file:
        file.write(audio)

    return audio