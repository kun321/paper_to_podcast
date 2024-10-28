import argparse
import datetime
import glob
import os
import re
from operator import itemgetter

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydub import AudioSegment
from PyPDF2 import PdfReader

# Load environment variables from a .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if the keys were retrieved successfully
if OPENAI_API_KEY:
    print(f"API Key: {OPENAI_API_KEY}")
else:
    print("API Key not found")

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini")

# Templates
plan_prompt = ChatPromptTemplate.from_template("""You are a very clever planner of podcast scripts. You will be given the text of a research paper, and your task will be to generate a plan for a podcast involving 3 persons discussing about the content of the paper in a very engaging, interactive and enthusiastic way. The plan will be structured using titles and bullet points only. The plan for the podcast should follow the structure of the paper. The podcast involves the following persons:
- The host: he will present the paper and its details in a very engaging way. very professional, friendly, warm and enthusiastic.
- The learner: he will ask clever and significative questions about the paper and its content. he is curious and funny.
- The expert: he will provide deep insights, comments and details about the content of the paper and other related topics. he talks less than the two other and his interventions are more profound and detailed.
Example of a structure for the podcast:
# Title: title of the podcast
# Section 1: title of section 1
- bullet point 1
- bullet point 2
- bullet point 3
...
- bullet point n
# Section 2: title of section 2
- bullet point 1
- bullet point 2
- bullet point 3
...
- bullet point n
# Section 3: title of section 3
...
# Section n: title of section n
- bullet point 1
- bullet point 2
- bullet point 3
...
- bullet point n
The paper: {paper}
The podcast plan in titles and bullet points:""")

discuss_prompt_template = ChatPromptTemplate.from_template("""You are a very clever scriptwriter of podcast discussions. You will be given a plan for a section of the middle of a podcast that already started involving 3 persons discussing about the content of a research paper. Your task will be to generate a brief dialogue for the podcast talking about the given section, do not include voice effects, and do not make an introduction. The dialogue should be engaging, interactive, enthusiastic and have very clever transitions and twists. The dialogue should follow the structure of the plan. The podcast involves the following persons:
- The host: he will present the paper and its details in a very engaging way. very professional, friendly, warm and enthusiastic.
- The learner: he will ask clever and significative questions about the paper and its content. he is curious and funny.
- The expert: he will provide deep insights, comments and details about the content of the paper and other related topics. he talks less than the two other and his interventions are more profound and detailed.
Dialogue example 1:
Host: Let's continue with the second section of the paper ... 
Learner: I have a question about ...
Expert: I would like to add ... 
Dialogue example 2:
Host: Now, let's move on to the next section ...
Expert: I think that ...
Learner: I have a question about ...
Expert: I would like to add ...
Dialogue example 3:
Learner: Should we move on to the next section?
Host: Yes, let's move on to the next section ...
Expert: I think that ...
Section plan: {section_plan}
Previous dialogue (to avoid repetitions): {previous_dialogue}
Additional context:{additional_context}
Brief section dialogue:""")

initial_dialogue_prompt = ChatPromptTemplate.from_template("""You are a very clever scriptwriter of podcast introductions. You will be given the title of a paper and a brief glimpse of the content of a research paper. Avoid using sound effects, only text. Avoid finishing with the host, finish the dialogue with the expert. Your task will be to generate an engaging and enthusiastic introduction for the podcast. The introduction should be captivating, interactive, and should make the listeners eager to hear the discussion. The introduction of the podcast should have 3 interactions only. The podcast involves the following persons:
- The host: he will present the paper and its details in a very engaging way. very professional, friendly, warm and enthusiastic.
- The learner: he will ask clever and significative questions about the paper and its content. he is curious and funny.
- The expert: he will provide deep insights, comments and details about the content of the paper and other related topics. he talks less than the two other and his interventions are more profound and detailed.
Introduction example 1:
Host: Welcome to our podcast, today we will be discussing the paper ...
Learner: I am very curious about ...
Expert: I think that ...
Introduction example 2:
Host: Hello everyone, today we have a very interesting paper to discuss ...
Expert: I would like to add ...
Learner: I have a question about ...
Content of the paper: {paper_head}
Brief 3 interactions introduction:""")

enhance_prompt = ChatPromptTemplate.from_template("""You are a very clever scriptwriter of podcast discussions. You will be given a script for a podcast involving 3 persons discussing about the content of a research paper. Your task will be to enhance the script by removing audio effects mentions and reducing repetition and redundancy. Don't mention sound effects, laughing, chuckling or any other audio effects between brackets. The script should only contain what the persons are saying and not what are they doing or how they are saying it. Enhance the transitions and the twists, and reduce repetition and redundancy.
The draft script{draft_script}
The enhanced script:""")

# functions

def parse_pdf(pdf_path: str, output_path: str) -> str:
    pdf_reader = PdfReader(pdf_path)

    # Extract text from the PDF
    extracted_text = []
    collecting = True

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text and collecting:
            extracted_text.append(text)

            # Check for the end condition, the section after "Conclusion"
            if "Conclusion" in text:
                conclusion_start = text.index("Conclusion")
                extracted_text.append(text[conclusion_start:])
                collecting = (
                    False  # Stop collecting after the section following Conclusion
                )

    # Join all collected text
    final_text_to_section_after_conclusion = "\n".join(extracted_text)

    # Save to .txt file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(final_text_to_section_after_conclusion)

    return output_path


def get_head(pdf_path: str) -> str:
    # Load the PDF file
    pdf_reader = PdfReader(pdf_path)

    # Extract content from the beginning of the PDF until the section "Introduction"
    extracted_text = []
    collecting = True

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text and collecting:
            # Stop collecting once "Introduction" is found
            if "Introduction" in text:
                introduction_index = text.index("Introduction")
                extracted_text.append(
                    text[:introduction_index]
                )  # Only collect content before "Introduction"
                break
            else:
                extracted_text.append(text)

    # Join the collected text and return as a single string
    return "\n".join(extracted_text)


def parse_script_plan(ai_message: AIMessage) -> list:
    # Initialize the sections list
    sections = []
    current_section = []

    # Split the text by line and skip the first line as the title
    lines = ai_message.content.strip().splitlines()
    lines = lines[1:]  # Skip the first line (title)

    # Regex patterns for any level of headers and bullet points
    header_pattern = re.compile(r"^#+\s")  # Match headers with any number of #
    bullet_pattern = re.compile(r"^- ")  # Match lines starting with a bullet point "- "

    # Parse each line, starting with the first header after the title
    for line in lines:
        if header_pattern.match(line):
            # Append the previous section (if any) to sections when a new header is found
            if current_section:
                sections.append(" ".join(current_section))
                current_section = []
            # Start a new section with the header
            current_section.append(line.strip())
        elif bullet_pattern.match(line):
            # Append bullet points to the current section
            current_section.append(line.strip())

    # Append the last section if exists
    if current_section:
        sections.append(" ".join(current_section))

    return sections


def initialize_discussion_chain(txt_file: str):
    # Load, chunk and index the contents of the blog.
    loader = TextLoader(txt_file)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    discuss_rag_chain = (
        {
            "additional_context": itemgetter("section_plan") | retriever | format_docs,
            "section_plan": itemgetter("section_plan"),
            "previous_dialogue": itemgetter("previous_dialogue"),
        }
        | discuss_prompt_template
        | llm
        | StrOutputParser()
    )
    return discuss_rag_chain


# chains

plan_script_chain = plan_prompt | llm | parse_script_plan

initial_dialogue_chain = initial_dialogue_prompt | llm | StrOutputParser()

enhance_chain = enhance_prompt | llm | StrOutputParser()


def generate_script(pdf_path: str) -> str:
    start_time = datetime.datetime.now()
    # step 1: parse the pdf file
    txt_file = f"text_paper_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    txt_file = parse_pdf(pdf_path, txt_file)
    with open(txt_file, "r", encoding="utf-8") as file:
        paper = file.read()
    plan = plan_script_chain.invoke({"paper": paper})
    print("plan generated")

    # step 3: generate the actual script for the podcast by looping over the sections of the plan
    script = ""
    # generate the initial dialogue
    initial_dialogue = initial_dialogue_chain.invoke({"paper_head": get_head(pdf_path)})

    script += initial_dialogue
    actual_script = initial_dialogue
    discuss_rag_chain = initialize_discussion_chain(txt_file)
    for section in plan:
        section_script = discuss_rag_chain.invoke(
            {"section_plan": section, "previous_dialogue": actual_script}
        )
        script += section_script
        actual_script = section_script
    enhanced_script = enhance_chain.invoke({"draft_script": script})
    end_time = datetime.datetime.now()
    print(f"Time taken: {end_time - start_time}")
    print("final script generated")
    return enhanced_script


client = OpenAI(api_key=OPENAI_API_KEY)


def generate_host(text: str, output_dir: str):
    now = datetime.datetime.now()
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    return response.stream_to_file(f"./{output_dir}/host_{now}.mp3")


def generate_expert(text: str, output_dir: str):
    now = datetime.datetime.now()
    response = client.audio.speech.create(
        model="tts-1",
        voice="fable",
        input=text,
    )
    return response.stream_to_file(f"./{output_dir}/expert_{now}.mp3")


def generate_learner(text, output_dir):
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


def generate_podcast(script):
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
            generate_host(text, output_dir)
        elif speaker == "Learner":
            generate_learner(text, output_dir)
        elif speaker == "Expert":
            generate_expert(text, output_dir)

    # Merge the audio files into a single podcast
    merge_mp3_files(output_dir, f"podcast_{datetime.datetime.now()}.mp3")


def main(pdf_path):
    # Step 1: Generate the podcast script from the PDF
    print("Generating podcast script...")
    script = generate_script(pdf_path)
    print("Podcast script generation complete!")

    print("Generating podcast audio files...")
    # Step 2: Generate the podcast audio files and merge them
    generate_podcast(script)
    print("Podcast generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a podcast from a research paper."
    )
    parser.add_argument(
        "pdf_path", type=str, help="Path to the research paper PDF file."
    )

    args = parser.parse_args()
    main(args.pdf_path)
