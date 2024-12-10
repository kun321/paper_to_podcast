import argparse
import os
from templates import enhance_prompt, initial_dialogue_prompt, plan_prompt
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
# from openai import OpenAI
from langchain_openai import ChatOpenAI
from utils.script import generate_script, parse_script_plan
from utils.audio_gen import generate_podcast

# Load environment variables from a .env file
load_dotenv()

# Retrieve the API keys from the environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# # Check if the keys were retrieved successfully
# if OPENAI_API_KEY and OPENROUTER_API_KEY:
#     print(f"API Keys retrieved successfully")
# else:
#     print("API Keys not found")

# # Initialize the OpenAI API client for audio generation
# client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize the ChatOpenAI model with OpenRouter
llm = ChatOpenAI(
    model="google/learnlm-1.5-pro-experimental:free",  
    openai_api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    max_tokens=40960,
)

# chains
chains = {
    "plan_script_chain": plan_prompt | llm | parse_script_plan,
    "initial_dialogue_chain": initial_dialogue_prompt | llm | StrOutputParser(),
    "enhance_chain": enhance_prompt | llm | StrOutputParser(),
}


def main(pdf_path):
    # Step 1: Generate the podcast script from the PDF
    print("Generating podcast script...")
    script = generate_script(pdf_path, chains, llm)
    print("Podcast script generation complete!")

    # Save the script to a text file
    script_filename = os.path.splitext(os.path.basename(pdf_path))[0] + "_script.txt"
    with open(script_filename, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"Script saved to {script_filename}")

    # print("Generating podcast audio files...")
    # # Step 2: Generate the podcast audio files and merge them
    # generate_podcast(script, client)
    # print("Podcast generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a podcast from a research paper."
    )
    parser.add_argument(
        "pdf_path", type=str, help="Path to the research paper PDF file."
    )

    args = parser.parse_args()
    main(args.pdf_path)
