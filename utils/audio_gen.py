from pydub import AudioSegment
import datetime
import os
import glob
import re


def generate_host(text: str, client, output_dir: str):
    now = datetime.datetime.now()
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    return response.stream_to_file(f"./{output_dir}/host_{now}.mp3")


def generate_expert(text: str, client, output_dir: str):
    now = datetime.datetime.now()
    response = client.audio.speech.create(
        model="tts-1",
        voice="fable",
        input=text,
    )
    return response.stream_to_file(f"./{output_dir}/expert_{now}.mp3")


def generate_learner(text, client, output_dir):
    now = datetime.datetime.now()
    response = client.audio.speech.create(
        model="tts-1",
        voice="echo",
        input=text,
    )
    return response.stream_to_file(f"./{output_dir}/learner_{now}.mp3")


def merge_mp3_files(directory_path, output_file):
    # Find all .mp3 files in the specified directory
    mp3_files = glob.glob(f"{directory_path}/*.mp3")

    # Sort files by datetime extracted from filename
    sorted_files = sorted(
        mp3_files,
        key=lambda x: datetime.datetime.strptime(
            re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", x).group(0),
            "%Y-%m-%d %H:%M:%S.%f",
        ),
    )

    # Initialize an empty AudioSegment for merging
    merged_audio = AudioSegment.empty()

    # Merge each mp3 file in sorted order
    for file in sorted_files:
        audio = AudioSegment.from_mp3(file)
        merged_audio += audio

    # Export the final merged audio
    merged_audio.export(output_file, format="mp3")
    print(f"Merged file saved as {output_file}")


def generate_podcast(script, client):
    # create a new directory to store the audio files
    output_dir = f"podcast_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.mkdir(output_dir)
    # Regex to capture "Speaker: Text"
    lines = re.findall(
        r"(Host|Learner|Expert):\s*(.*?)(?=(Host|Learner|Expert|$))", script, re.DOTALL
    )

    for speaker, text, _ in lines:
        # Strip any extra spaces or newlines
        text = text.strip()

        # Direct the text to the appropriate function
        if speaker == "Host":
            generate_host(text, client, output_dir)
        elif speaker == "Learner":
            generate_learner(text, client, output_dir)
        elif speaker == "Expert":
            generate_expert(text, client, output_dir)

    # Merge the audio files into a single podcast
    merge_mp3_files(output_dir, f"podcast_{datetime.datetime.now()}.mp3")
