"""
Robo-ly: RL Skill Learning Interface

This is an interactive chat interface for learning and executing robotic skills
using reinforcement learning. It interfaces with an LLM to determine whether to
train new skills, fine-tune existing skills, or execute known skills.
"""

import json
import os
import re
import signal
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
import traceback

import IPython
import json5
import yaml

import isaacgym

print(isaacgym)
from datetime import datetime

from isaacgymenvs.tasks import isaacgym_task_map
from openai import OpenAI

warnings.filterwarnings("ignore")


def upper_camel_to_snake_case(text):
    """
    Converts an Upper Camel Case string to snake_case.
    """
    # Insert underscore before any uppercase letter that follows a lowercase letter
    s1 = re.sub(r"([a-z])([A-Z])", r"\1_\2", text)
    # Insert underscore before any uppercase letter that follows multiple uppercase letters
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    return s2.lower()


class Config:
    """Simple yaml config wrapper"""

    def __init__(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, "r") as file:
            self.dict = yaml.safe_load(file) or {}

    def get(self, *keys, default=None):
        """Get nested config values."""
        value = self.dict
        for key in keys:
            if isinstance(value, dict):
                if key not in value:
                    return default
                value = value[key]
            else:
                return default
        return value


class SkillLibrary:
    """Manager for the skill library."""

    def __init__(self, library_path: str):
        """Load skill library from YAML file."""
        self.library_path = library_path
        self.skills_dict = Config(library_path).dict

    def get_skill(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Get a skill from the library."""
        return self.skills_dict.get(skill_name)

    def add_skill(self, skill_name: str, skill_data: Dict[str, Any]):
        """Add or update a skill in the library."""
        self.skills_dict[skill_name] = skill_data

    def save(self):
        """Save the skill library to disk."""
        with open(self.library_path, "w") as file:
            yaml.dump(self.skills_dict, file, default_flow_style=False)

    def __str__(self):
        """Return string representation of skill library."""
        return str(self.skills_dict)


class ResponseParser:
    """Parser for LLM responses containing JSON."""

    @staticmethod
    def parse_json(response_text: str) -> Optional[Dict]:
        """Parse JSON from response text, handling markdown code blocks."""
        # Try to extract JSON from markdown code block
        json_pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, response_text, re.DOTALL)

        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        try:
            start_index = response_text.find("{")
            if start_index == -1:
                return None

            decoder = json.JSONDecoder()
            parsed_obj, _ = decoder.raw_decode(response_text[start_index:])
            return parsed_obj
        except json.JSONDecodeError:
            return None

    @staticmethod
    def robust_parse_json(response_text: str) -> Optional[Dict]:
        """Parse JSON with fallback to json5 for lenient parsing."""
        # Sanitize input
        sanitized_text = response_text.replace("\u00a0", " ")

        # Try strict parsing first
        parsed_data = ResponseParser.parse_json(sanitized_text)
        if parsed_data:
            return parsed_data

        # Fall back to json5 for lenient parsing
        start_index = -1
        first_brace = sanitized_text.find("{")
        first_bracket = sanitized_text.find("[")

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start_index = first_brace
        elif first_bracket != -1:
            start_index = first_bracket

        if start_index != -1:
            try:
                return json5.loads(sanitized_text[start_index:])
            except Exception:
                return None
        return None


class TrainingManager:
    """Manager for training processes and skill execution."""

    def __init__(self, config: Config, skill_library: SkillLibrary):
        """Initialize the training manager."""
        self.config = config
        self.skill_library = skill_library
        self.training_process = {
            "process": None,
            "name": None,
            "description": None,
            "environment": None,
            "log_file": None,
        }
        self.visual_process = None

    def run_process(self, cmd: str, log_name: str):
        """
        Run a command with output redirected to a log file.

        Args:
            cmd: The command to run
            log_name: Name for the log file (without extension)

        Returns:
            Tuple of (subprocess.Popen object, log filename)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{timestamp}_{log_name}.log"

        print(f'Running code... (see logs in "{log_filename}")')

        with open(log_filename, "w") as log_file:
            # Write the command to the log file first
            log_file.write(f"Command: {cmd}\n")
            log_file.flush()

            process = subprocess.Popen(
                cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp
            )

        return process, log_filename

    def start_visual_demo(self, task: str):
        """Start a visual demo process."""
        cmd = self.config.get("training", "test_command").format(task=task, checkpoint="").replace("checkpoint= ", "")
        self.visual_process, _ = self.run_process(cmd, f"visual_demo_{task}")

    def stop_visual_demo(self):
        """Stop the current visual demo process."""
        if self.visual_process is not None:
            try:
                os.killpg(os.getpgid(self.visual_process.pid), signal.SIGINT)
            except Exception:
                pass
            self.visual_process = None

    def execute_skill(self, skill_name: str):
        """Execute an existing skill from the library by showing its training video."""
        skill = self.skill_library.get_skill(skill_name)
        if not skill:
            print(f"⚠️ Skill '{skill_name}' not found in library.")
            return

        # Friendly message that we know this skill
        print(f"✓ Ok, I know how to do that!")

        # OLD: This would run the policy in real-time but doesn't capture videos
        # headless = "False" if not self.config.get("subprocess", "visual_headless") else "True"
        # force_render = "True" if self.config.get("subprocess", "visual_force_render") else "False"
        # cmd = self.config.get("training", "test_command").format(
        #     task=skill["environment"], checkpoint=skill["policy_path"], headless=headless, force_render=force_render
        # )
        # self.stop_visual_demo()
        # self.visual_process, _ = self.run_process(cmd, f"execute_{skill_name}")

        # NEW: Find and display the training video instead
        # Find the training video for this skill by looking in the policy's parent directories
        policy_path = skill.get("policy_path")
        if not policy_path:
            print(f"⚠️ No policy path found for skill '{skill_name}'")
            return

        # Extract the datetime folder from the policy path
        # e.g. .../eureka/2025-11-13_00-00-31/policy-.../runs/.../*.pth
        match = re.search(r"eureka/([^/]+)", policy_path)
        if not match:
            print(f"⚠️ Could not find eureka folder in policy path: {policy_path}")
            return

        eureka_base = Path(policy_path[:policy_path.index(match.group(0)) + len(match.group(0))])

        # Search for video files
        video_files = list(eureka_base.glob("**/*.mp4"))

        if video_files:
            # Sort by modification time and take the most recent
            video_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            video_path = str(video_files[0])
            print(f"🎥 Training video for '{skill_name}': {video_path}")

            # Optionally open the video with the default system viewer
            if self.config.get("subprocess", "auto_open_videos", default=False):
                try:
                    if os.name == "posix":  # Linux/Mac
                        if os.uname().sysname == "Darwin":  # macOS
                            subprocess.Popen(["open", video_path])
                        else:  # Linux
                            subprocess.Popen(["xdg-open", video_path])
                    elif os.name == "nt":  # Windows
                        os.startfile(video_path)
                    print("📺 Opening video in default viewer...")
                except Exception as e:
                    print(f"⚠️ Could not auto-open video: {e}")
        else:
            print(f"⚠️ No training videos found for skill '{skill_name}' in {eureka_base}")

    def train_skill(self, task: str, skill_name: str, description: str, base_skill: Optional[str] = None):
        """Train a new skill or fine-tune an existing one."""
        task = task.strip("'\"")

        # Friendly message that we need to learn this skill
        print("Ugh, I will need to learn that...")

        # If base_skill is provided, append its reward code to the description
        full_description = description
        if base_skill:
            base_skill_data = self.skill_library.get_skill(base_skill)
            if base_skill_data and base_skill_data.get("reward_code_path"):
                reward_code_path = base_skill_data["reward_code_path"]
                if os.path.exists(reward_code_path):
                    try:
                        with open(reward_code_path, "r") as f:
                            reward_code = f.read()
                        full_description = (
                            f"{description}\n\nPrevious reward function code from {base_skill} to build upon:\n"
                            f"```python\n{reward_code}\n```"
                        )
                    except Exception as e:
                        print(f"⚠️ Could not read reward code from {reward_code_path}: {e}")
                else:
                    print(f"⚠️ Reward code path does not exist: {reward_code_path}")
            else:
                print(f"⚠️ Base skill '{base_skill}' not found or has no reward code")

        # Write description to a temporary file to avoid shell escaping issues
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        desc_filename = f"{timestamp}_train_{skill_name}_description.txt"
        with open(desc_filename, "w") as f:
            f.write(full_description)

        # Build command with description file path (use absolute path)
        desc_filepath = os.path.abspath(desc_filename)
        train_cmd_template = self.config.get("training", "train_command")
        env_snake = upper_camel_to_snake_case(task)
        cmd = train_cmd_template.replace("{env}", env_snake)
        cmd = cmd.replace('"{description}"', desc_filepath)

        # Use run_process to handle logging
        process, log_filename = self.run_process(cmd, f"train_{skill_name}")

        self.training_process["process"] = process
        self.training_process["name"] = skill_name
        self.training_process["environment"] = task
        self.training_process["description"] = description
        self.training_process["log_file"] = log_filename  # Wait for training to complete

        while process.poll() is None:
            pass

        # Remove temporary description file
        if os.path.exists(desc_filename):
            os.remove(desc_filename)

        # Check training status and update skill library if complete.

        # Check if process exited with an error (log file contents)
        log_file = self.training_process.get("log_file")
        has_errors = False
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    log_content = f.read()
                    # Look for common error indicators
                    if ("Traceback" in log_content and "Error" in log_content) or \
                       ("Could not find" in log_content and "Available options" in log_content):
                        has_errors = True
            except Exception:
                pass

        if has_errors:
            error_msg = f"❌ Skill {self.training_process['name']} training failed. Check the log file {log_file} for details."
            print(error_msg)
            print("🔄 Retrying...")
            # Retry the training once
            process, log_filename = self.run_process(cmd, f"train_{skill_name}_retry")
            self.training_process["process"] = process
            self.training_process["log_file"] = log_filename
            while process.poll() is None:
                pass
            # Check again if it failed
            if os.path.exists(log_filename):
                try:
                    with open(log_filename, "r") as f:
                        log_content = f.read()
                        if ("Traceback" in log_content and "Error" in log_content) or \
                           ("Could not find" in log_content and "Available options" in log_content):
                            error_msg = f"❌ Skill {self.training_process['name']} training failed again after retry. Check the log file {log_filename} for details."
                            print(error_msg)
                            return error_msg
                except Exception:
                    pass
            # Update log_file to the retry log
            log_file = log_filename

        # Training is complete, find the policy file
        base_path = Path(self.config.get("paths", "eureka_output_base"))

        if not base_path.exists():
            return f"Skill {self.training_process['name']} training complete, but output path not found."

        # Find the specific policy file with the naming pattern: {task}_roboly_{skill_name}.pth
        env_name = self.training_process["environment"]
        pth_files = list(base_path.glob(f"**/runs/{env_name}*/**/*{env_name}GPT*.pth"))

        if not pth_files:
            return f"Skill {self.training_process['name']} training complete, but no policy file found."

        # Sort by modification time and take the most recent one
        pth_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        policy_path = str(pth_files[0])

        # Find the reward function file in the same datetime base folder
        # e.g. .../eureka/2025-11-11_12-14-55/**/*_rewardonly.py
        match = re.search(r"eureka/[^/]+", policy_path)
        datetime_folder = None
        if match:
            eureka_base = policy_path[: match.end()]
            datetime_folder = Path(eureka_base)
        else:
            print(f"⚠️  Could not extract datetime folder from policy path: {policy_path}")

        reward_code_path = ""
        reward_files = list(datetime_folder.glob("**/*_rewardonly.py"))
        if reward_files:
            # Sort by modification time and take the most recent
            reward_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            reward_code_path = str(reward_files[0])
        else:
            print(f"⚠️  No reward code file found in {datetime_folder}")

        # Add the new skill to the library
        self.skill_library.add_skill(
            self.training_process["name"],
            {
                "policy_path": policy_path,
                "policy_description": self.training_process["description"],
                "reward_code_path": reward_code_path,
                "environment": self.training_process["environment"],
            },
        )

        # Save the updated library
        self.skill_library.save()

        print(f"Skill {self.training_process['name']} training is complete.")

        # Look for generated video files
        if datetime_folder:
            video_files = list(datetime_folder.glob("**/*.mp4"))
            if video_files:
                # Sort by modification time and take the most recent
                video_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                video_path = str(video_files[0])
                print(f"🎥 Training video available at: {video_path}")

                # Optionally open the video with the default system viewer
                if self.config.get("subprocess", "auto_open_videos", default=False):
                    try:
                        if os.name == "posix":  # Linux/Mac
                            if os.uname().sysname == "Darwin":  # macOS
                                subprocess.Popen(["open", video_path])
                            else:  # Linux
                                subprocess.Popen(["xdg-open", video_path])
                        elif os.name == "nt":  # Windows
                            os.startfile(video_path)
                        print("📺 Opening video in default viewer...")
                    except Exception as e:
                        print(f"⚠️  Could not auto-open video: {e}")
            else:
                print(f"ℹ️  No video files found in {datetime_folder}")

    def delete_skill(self, skill_name: str):
        """Delete a skill from the library."""
        if skill_name in self.skill_library.skills_dict:
            del self.skill_library.skills_dict[skill_name]
            self.skill_library.save()
            print(f"🗑️  Deleted skill: {skill_name}")
        else:
            print(f"⚠️ Skill '{skill_name}' not found in library.")


class ChatInterface:
    """Interactive chat interface for the RL skill learning system."""

    def __init__(self, config: Config):
        """Initialize the chat interface."""
        self.config = config
        self.skill_library = SkillLibrary(config.get("paths", "skill_library"))
        self.training_manager = TrainingManager(config, self.skill_library)
        self.client = None
        self.messages = []

        # Create log file for LLM responses
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{timestamp}_llm_responses.log"

    def initialize_llm(self):
        """Initialize the OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        return True

    def log_response(self, user_prompt: str, robo_response: str):
        """Log the user prompt and LLM response to the log file."""
        with open(self.log_filename, "a") as log_file:
            log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
            log_file.write(f"\n👤: {user_prompt}\n")
            log_file.write(f"\n🤖: {robo_response}\n")
            log_file.write(f"{'-' * 20}\n")

    def format_system_prompt(self) -> str:
        """Load and format the system prompt."""
        prompt_path = self.config.get("paths", "system_prompt")

        with open(prompt_path, "r") as file:
            prompt = file.read()

        prompt = prompt.replace("{skill_library}", str(self.skill_library))
        prompt = prompt.replace("{training_environments}", str(list(isaacgym_task_map.keys())))

        print(prompt)
        return prompt

    def initialize_conversation(self):
        """Initialize the conversation with system prompt."""
        system_prompt = self.format_system_prompt()
        self.messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

    def process_llm_response(self, response: str):
        """Process the parsed LLM response and take appropriate action(s)."""
        parsed_response = ResponseParser.robust_parse_json(response)

        if parsed_response is None:
            # non-json plain text, probably conversational
            print(response)
            return

        actions = parsed_response.get("actions", [])
        if not actions:
            return

        for action in actions:
            action_type = action.get("action")

            if action_type == "show":
                # Execute existing skill
                skill_name = action.get("skill_name")
                if skill_name:
                    self.training_manager.execute_skill(skill_name)

            elif action_type == "train":
                # Train new skill
                task = action.get("task")
                skill_name = action.get("skill_name")
                description = action.get("description")
                base_skill = action.get("base_skill")

                if task and skill_name and description:
                    self.training_manager.train_skill(task, skill_name, description, base_skill)

            elif action_type == "delete":
                # Delete existing skill
                skill_name = action.get("skill_name")
                if skill_name:
                    self.training_manager.delete_skill(skill_name)

    def run(self):
        """Run the main chat loop."""
        if not self.initialize_llm():
            return

        self.initialize_conversation()

        print("""
   ___       __         __    
  / _ \___  / /  ___   / /_ __
 / , _/ _ \/ _ \/ _ \ / / // /
/_/|_|\___/_.__/\___//_/\_, / 
                       /___/  """)
        print("(Type 'exit' or 'quit' to end the conversation.)")
        print("(Type 'debug' to open interactive debugging.)")
        print("-" * 40)
        print("🤖 Ugh, what do you want?")

        # Main chat loop
        while True:
            # Get user input
            user_prompt = input("👤 > ").strip()

            # Check for exit commands
            if user_prompt.lower() in ["exit", "quit"]:
                print("👋 Thank god!")
                break

            if user_prompt.lower() in ["debug", "ipython"]:
                IPython.embed()
                continue

            # Append user message to history
            self.messages.append({"role": "user", "content": user_prompt})

            try:
                print("\n🤖 Thinking...")

                # Send conversation to API
                model = self.config.get("llm", "model", default="gpt-4o-mini")
                chat_completion = self.client.chat.completions.create(messages=self.messages, model=model)

                # Extract assistant's response
                robo_response = chat_completion.choices[0].message.content

                # Log the conversation to file instead of printing
                self.log_response(user_prompt, robo_response)

                # Parse and process response
                self.process_llm_response(robo_response)

                # Append assistant's response to history
                self.messages.append({"role": "assistant", "content": robo_response})

            except Exception as e:
                # Error handling
                print(f"\n❌ An error occurred: {e}")
                traceback.print_exc()
                # Remove the last user message to allow retrying
                self.messages.pop()


def main():
    config = Config("config.yml")

    interface = ChatInterface(config)
    interface.run()


if __name__ == "__main__":
    main()
