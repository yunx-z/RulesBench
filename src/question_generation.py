import os
import json
import re
from tqdm import tqdm

from src.LLM import complete_text

MODEL = "gemini-2.0-pro-exp-02-05"

def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from the given text.
    """
    clean = re.compile(r'<[^>]+>')
    return re.sub(clean, '', text)

def count_lines(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def extract_json(response):
    match = re.search(r'```json(.*?)```', response, re.DOTALL)
    if match:
        code_content = match.group(1).strip()
        return code_content
    else:
        return response

def safe_load_json(response):
    output = extract_json(response)
    if not output:
        return {}
    try:
        data = json.loads(output)
    except Exception as e:
        print(e)
        print("Error parsing JSON from response:", output)
        data = {}
    return data

def validate_mcq_schema(mcq: dict, include_citation: bool = True) -> bool:
    """
    Check that the mcq dictionary contains all the required keys.
    
    Expected keys are:
      "Question", "A", "B", "C", "D", "Correct Answer", and "Explanation"
    """
    required_keys = ["Question", "A", "B", "C", "D", "Correct Answer", "Explanation"]
    if include_citation:
        required_keys.append("Citation")
    if not isinstance(mcq, dict):
        return False
    return all(key in mcq for key in required_keys)

def generate_mcq(formatted_question: str, formatted_answer: str, rulebook_text: str = None, full_content: list = None, max_retry: int = 5):
    """
    Generate a complete multiple-choice question with optional context.
    
    Parameters:
      - formatted_question: The base question.
      - formatted_answer: The correct answer.
      - rulebook_text: (Optional) Additional context from the rulebook.
      - full_content: (Optional) Forum discussion posts (list of dicts with a "content" key).
      - max_retry: Maximum number of retries if the generated JSON does not match the schema.
      
    The expected output JSON format is:
    
    {
      "Question": "<multiple-choice question text>",
      "A": "<option A>",
      "B": "<option B>",
      "C": "<option C>",
      "D": "<option D>",
      "Correct Answer": "<the correct option letter>",
      "Explanation": "<detailed explanation>",
      "Citation": "<reference to the rulebook section or forum discussion>"
    }
    
    Example:
    {
      "Question": "In the board game Troyes, ...?",
      "A": "Yes – if you have the available actions, ...",
      "B": "Yes – you can place two meeples on the same card, ...",
      "C": "No – the rules allow only one meeple per activity card per round, ...",
      "D": "No – you can only place a meeple on an activity card if no other meeple has been placed...",
      "Correct Answer": "C",
      "Explanation": "In Troyes, each activity card can only host one meeple per round..."
    }
    
    """
    prompt = (
        "Can you make this question into a multiple-choice question? We are building a benchmark for LLM on boardgames rules question answering so want some hard question.\n\n"
        f"Question:\nThis question is about the rules of a boardgame called Pax Renaissance (2nd Edition). {formatted_question}\n\n"
        f"Answer:\n{formatted_answer}\n\n"
    )
    
    prompt_file = "data/prompt"
    if rulebook_text:
        prompt += f"Please come up with distractors based on the provided rulebook below:\n{rulebook_text}\n\n"
        prompt_file += "_rulebook"
        
    if full_content and len(full_content) >= 3:
        forum_posts = []
        for post in full_content:
            content = post.get("content", "")
            cleaned = strip_html_tags(content)
            a_post = f"username: {post['username']}\tpost_date: {post['post_date']}\ncontent: {cleaned}"
            forum_posts.append(a_post)
        forum_text = "\n\n".join(forum_posts)
        prompt += f"Please come up with distractors based on the BGG forum discussion below:\n{forum_text}\n\n"
        prompt_file += "_forum"

    include_citation = rulebook_text or full_content
   
    if include_citation:
        prompt += (
            "Output the result as a JSON dictionary with the following keys: \"Question\", \"A\", \"B\", \"C\", \"D\", \"Correct Answer\", \"Explanation\", and \"Citation\".\n"
            "The 'Citation' field should provide a reference to the rulebook section or the forum discussion, depending on the source used.\n"
            "For example:\n"
            "{\n"
            '  "Question": "In the board game Troyes, ...?",\n'
            '  "A": "Yes – if you have the available actions, ...",\n'
            '  "B": "Yes – you can place two meeples on the same card, ...",\n'
            '  "C": "No – the rules allow only one meeple per activity card per round, ...",\n'
            '  "D": "No – you can only place a meeple on an activity card if no other meeple has been placed...",\n'
            '  "Correct Answer": "C",\n'
            '  "Explanation": "In Troyes, each activity card can only host one meeple per round...",\n'
            '  "Citation": "Rulebook, Section 3.2: Worker Placement; username and post date for forum discussion"\n'
            "}\n\n"
            "Do not include any additional text."
        )
    else:
        prompt += (
            "Output the result as a JSON dictionary with the following keys: \"Question\", \"A\", \"B\", \"C\", \"D\", "
            "\"Correct Answer\", and \"Explanation\".\n"
            "For example:\n"
            "{\n"
            '  "Question": "In the board game Troyes, ...?",\n'
            '  "A": "Yes – if you have the available actions, ...",\n'
            '  "B": "Yes – you can place two meeples on the same card, ...",\n'
            '  "C": "No – the rules allow only one meeple per activity card per round, ...",\n'
            '  "D": "No – you can only place a meeple on an activity card if no other meeple has been placed...",\n'
            '  "Correct Answer": "C",\n'
            '  "Explanation": "In Troyes, each activity card can only host one meeple per round..."\n'
            "}\n\n"
            "Do not include any additional text."
        )
    

    prompt_file += ".txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, 'w') as f:
            f.write(prompt)
    
    retries = 0
    while retries < max_retry:
        output = complete_text(prompt, model=MODEL)
        mcq = safe_load_json(output)
        if validate_mcq_schema(mcq, include_citation=include_citation):
            return mcq
        retries += 1
        print(f"Retry {retries}/{max_retry} for schema validation...")
    
    # Return the last attempt even if it doesn't fully match, or handle as needed.
    return mcq


def process_examples(examples_path: str, rulebook_path: str, output_path: str):
    """
    Process each QA pair from the examples file and generate four versions of a complete multiple-choice question:
      - from_scratch: Using only the question and answer.
      - from_rulebook: Using the rulebook text as context.
      - from_forum: Using forum discussion as context (if there are at least three posts).
      - from_rulebook_and_forum: Using both the rulebook text and forum discussion (if there are at least three posts).
      
    The final output for each example is a JSON object with the generated MCQs and the original URL.
    """
    processed_examples = []
    total_lines = count_lines(examples_path)
    print(f"Saving to {output_path}")
    
    # Load the rulebook text.
    with open(rulebook_path, "r", encoding="utf-8") as f:
        rulebook_text = f.read()
    
    with open(examples_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing questions", total=total_lines):
            example = json.loads(line)
            formatted_question = example["formatted_question"]
            formatted_answer = example["formatted_answer"]
            full_content = example.get("full_content", [])
            
            generated_mcq = {}
            generated_mcq["from_scratch"] = generate_mcq(formatted_question, formatted_answer)
            generated_mcq["from_rulebook"] = generate_mcq(formatted_question, formatted_answer, rulebook_text=rulebook_text)
            
            if len(full_content) >= 3:
                generated_mcq["from_forum"] = generate_mcq(formatted_question, formatted_answer, full_content=full_content)
                generated_mcq["from_rulebook_and_forum"] = generate_mcq(formatted_question, formatted_answer, rulebook_text=rulebook_text, full_content=full_content)
            
            final_output = {
                "url": example["url"],
                "generated_mcq": generated_mcq,
            }
            processed_examples.append(final_output)
    
            # Save line-delimited JSON output.
            with open(output_path, 'a', encoding="utf-8") as writer:
                writer.write(json.dumps(final_output) + '\n')
    
    # Save pretty JSON version.
    with open(output_path.replace("jsonl", "json"), 'w', encoding="utf-8") as writer:
        json.dump(processed_examples, writer, indent=2)

def main():
    rulebook_path = "rules_material/pax_ren_2e/paxren_rulebook1.txt"
    examples_path = "data/paxren_100_hot.jsonl"
    output_path = "data/paxren_100_hot_mcq.jsonl"

    process_examples(examples_path, rulebook_path, output_path)

if __name__ == "__main__":
    main()

