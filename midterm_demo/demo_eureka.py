import os
from openai import OpenAI
import yaml
import json
import sys
import re
import subprocess
from pathlib import Path
import traceback
import time
import signal
import isaacgym
import shlex
import json5 # Import the new library
from isaacgymenvs.tasks import isaacgym_task_map

import warnings
warnings.filterwarnings('ignore')

def format_system_prompt(prompt_path:str, skill_library:dict):

    with open(prompt_path, 'r') as file:
        prompt = file.read()

    prompt = prompt.replace('{skill_library}', str(skill_library))
    prompt = prompt.replace('{training_environments}', str(list(isaacgym_task_map.keys())))
    print(prompt)

    return prompt


def get_training_status(training_process_dict, skill_library):

    training_status = ''
    if training_process_dict["process"] is not None:
        if training_process_dict["process"].poll() is None:
            training_status = f"Skill {training_process_dict['name']} training in progress."
        else:
            base_path = Path('/home/ttr/Eureka/eureka/outputs/eureka')

            # Get all subdirectories
            subdirs = [d for d in base_path.iterdir() if d.is_dir()]
            subdirs.sort()
            last_subdir = subdirs[-1]
            
            print(f"Last subdirectory: {last_subdir}")
            
            # Find all .pth files in that subdirectory
            pth_files = list(last_subdir.glob(f'**/*.pth'))  # ** for recursive search

            skill_library[training_process_dict["name"]] = {
                'policy_path': pth_files[-1],
                'policy_description': training_process_dict["description"],
                'previous_reward_function': "path/to/reward_function.py",
                'environment': training_process_dict["environment"]
            }

            training_status = f"Skill {training_process_dict['name']} training is complete. New skill library state: {skill_library}"
    print(f'Training status: {training_status}')
    return training_status

def parse_response(response_text: str):
    """
    Parses a JSON object from a string, whether it's in a markdown code block or not.
    
    This function is designed to be robust against common LLM output formats.
    
    Args:
        response_text: The text containing the JSON object.

    Returns:
        A Python dictionary or list if JSON is found, otherwise None.
    """
    # 1. First, try to extract JSON from a markdown code block
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, response_text, re.DOTALL)
    
    # If a match is found in a code block, try to parse it
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If parsing fails, we'll proceed to the more general search
            pass

    # 2. If no code block or parsing failed, search for the first '{' and try to parse
    try:
        # Find the first occurrence of '{' which marks the potential start of a JSON object
        start_index = response_text.find('{')
        if start_index == -1:
            return None # No JSON object found

        # Use the JSON decoder to parse from that point.
        # raw_decode is perfect for this as it will parse one valid JSON object
        # and return the index where it stopped.
        decoder = json.JSONDecoder()
        parsed_obj, end_index = decoder.raw_decode(response_text[start_index:])
        return parsed_obj

    except json.JSONDecodeError:
        # This will catch errors if the string starting with '{' is not valid JSON
        return None

def robust_parse_response(response_text: str):
    """
    A more robust version that handles common JSON errors like trailing commas.
    """
    # First, sanitize the input to remove common non-standard characters
    # This fixes the non-breaking space issue from your previous question!
    sanitized_text = response_text.replace('\u00A0', ' ')

    # Try the strict Python json parser first
    parsed_data = parse_response(sanitized_text) # Calls the function from above
    if parsed_data:
        return parsed_data
        
    # If standard parsing fails, try the more lenient json5 parser
        # Find the first '{' or '[' to start parsing from
    start_index = -1
    first_brace = sanitized_text.find('{')
    first_bracket = sanitized_text.find('[')
    
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_index = first_brace
    elif first_bracket != -1:
        start_index = first_bracket

    if start_index != -1:
        return json5.loads(sanitized_text[start_index:])
    return None
    

def call_tools(parsed_response, skill_library, visual_process, training_process_dict):
    """
    Return string to feed back into the LLM
    """
    if parsed_response is None:
        return
    
    if not parsed_response['train']:
        # Load known skill

        skill_name = parsed_response['chosen_skill']
        cmd = f"python train.py task={skill_library[skill_name]['environment']} test=True checkpoint={skill_library[skill_name]['policy_path']} headless=False force_render=True"
        process = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp)  
                
        if visual_process is not None:
            os.killpg(os.getpgid(visual_process.pid), signal.SIGINT)
        visual_process = process
        
    else:
        # Train a new skill
        task_name = parsed_response['training_args']['task']
        task_description = parsed_response['training_args']['description']
        print(task_name)
        task_name = task_name.strip("'\"")


        python_path = sys.executable
    
        # 1. Define the command to run
        cmd = f"{python_path} ../../eureka.py env.description=\"{task_description}\" task={task_name}"

        # 2. Create the full script for bash to execute.
        #    This now includes the commands to initialize and activate Conda.

        process = subprocess.Popen([cmd], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp)

        training_process_dict['process'] = process
        training_process_dict['name'] = parsed_response['training_args']['skill_name']  
        training_process_dict['environment'] = task_name
        training_process_dict['description'] = parsed_response['training_args']['description']

        while process.poll() is None:
            pass
        
    return
    


def main():
    """
    Main function to run an interactive chat session with the OpenAI API.
    """
    try:
        # --- API Key and Client Initialization ---
        # Retrieve the API key from environment variables for security.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: The OPENAI_API_KEY environment variable is not set.")
            print("Please set it to your OpenAI API key.")
            return

        client = OpenAI(api_key=api_key)

        with open('/home/ttr/Eureka/roboly/skill_library.yaml', 'rb') as file:
            skill_library = yaml.safe_load(file)



        cmd = f"python train.py task=CartpoleSpin test=True headless=False force_render=True"
        visual_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp)  
        training_process_dict = {"process":None, "name":None, "description":None, "environment":None}

        # --- Conversation History ---
        # Initialize the conversation with a system message to set the assistant's behavior.
        system_prompt = format_system_prompt('/home/ttr/Eureka/roboly/demo_prompt.txt', skill_library=skill_library)
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        print("🤖 Ugh, what do you want?")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("-" * 40)

        # --- Interactive Chat Loop ---
        while True:
            # Get user input
            user_prompt = input("👤 You: ")

            # Check for exit commands
            if user_prompt.lower() in ["exit", "quit"]:
                print("👋 Thank god!")
                break
            
            # Append user message to the history
            messages.append({"role": "user", "content": user_prompt})

            try:
                # --- API Call ---
                # Show a waiting message
                print("\n🤖 Thinking...")
                
                # Send the entire conversation history to the API
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    # model="gpt-4.1-mini", # You can change this to "gpt-3.5-turbo" or other models.
                    model="gpt-5"
                )

                # --- Process and Print Response ---
                # Extract the assistant's reply
                assistant_response = chat_completion.choices[0].message.content
                print(f"\n🤖 Assistant:\n{assistant_response}\n")

                # Append assistant's response to the history for context in the next turn

                parsed_response = robust_parse_response(assistant_response)
                print(parsed_response)

                call_tools(parsed_response, skill_library=skill_library, visual_process=visual_process, training_process_dict=training_process_dict)
                
                if parsed_response is not None and parsed_response['train']:
                    training_status_update = get_training_status(training_process_dict, skill_library)
                    assistant_response += training_status_update    

                messages.append({"role": "assistant", "content": assistant_response})



            except Exception as e:
                # --- Error Handling ---
                print(f"\n❌ An error occurred: {e}")
                # Remove the last user message if the API call failed, to allow retrying.
                messages.pop()

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n👋 Thank god!")
    except Exception as e:
        traceback.print_exc()
        print(f"\n❌ A critical error occurred: {e}")

if __name__ == "__main__":
    main()