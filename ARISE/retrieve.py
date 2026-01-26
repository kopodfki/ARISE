import requests
import os
import traceback

import setGPU
import os
import pickle
import re
import openai
import json
import time
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from os import error, path as osp
from tqdm import tqdm
import scenic
from scenic.simulators import carla
from scenic.core.simulators import SimulationCreationError

local_path = osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__))))
DB = osp.join(local_path, 'ARISE', 'db', 'database_v1_scenic3.pkl')

# Constants for API call retries
MAX_API_RETRIES = 5      # Maximum number of retry attempts
RETRY_DELAY_SECONDS = 4  # Initial delay between retries
RETRY_BACKOFF_FACTOR = 2  # Multiply delay by this factor after each retry

client = 0
iterations = 0
topk = 0
llm_model = ""
top_behavior_descriptions, top_geometry_descriptions, top_spawn_descriptions, top_misc_descriptions, top_weather_descriptions, current_requirements = [], [], [], [], [], []
top_behavior_snippets, top_geometry_snippets, top_spawn_snippets, top_misc_snippets, top_weather_snippets = [], [], [], [], []
current_behavior, current_geometry, current_spawn, current_misc, current_weather = "", "", "", "", ""

# stores tuples of (file_path, success), to then return to the main script
generated_scenarios = []


def make_llm_api_call(model_name, system_prompt, user_prompt, client, max_retries=MAX_API_RETRIES, reason=False):
    """
    Make an API call to the specified LLM with retry logic for handling service outages.

    Args:
        model_name: The name of the LLM model to use
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        client: The client object (OpenAI or Google Gemini)
        max_retries: Maximum number of retry attempts

    Returns:
        The response content from the LLM
    """
    retry_count = 0
    delay = RETRY_DELAY_SECONDS

    while retry_count <= max_retries:
        try:
            if model_name.startswith('gpt'):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    top_p=1
                )
                return response.choices[0].message.content

            elif model_name.startswith('gemini') or model_name.startswith('gemma'):
                if not reason:
                    response = client.models.generate_content(
                        model=model_name,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=1.0),
                        contents=user_prompt,
                    )
                else:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.3,
                            thinking_config=types.ThinkingConfig(
                                thinking_budget=2048
                            )
                        )
                    )
                return response.text

            elif model_name.startswith('deepseek'):
                deepseek_url = "http://172.16.59.199:11434/api/generate"
                payload = {
                    "model": "deepseek-v3",
                    'prompt': user_prompt,
                    "stream": False,
                    "temperature": 1.0
                }

                response = requests.post(deepseek_url, json=payload)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses

                response_data = response.json()
                return response_data['response']

        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                error_message = str(e)
                print(f"API call failed with error: {error_message}")

                # Check if it's a 503 error or other retriable error
                if "503" in error_message or "UNAVAILABLE" in error_message:
                    print(
                        f"Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                    delay *= RETRY_BACKOFF_FACTOR  # Exponential backoff
                else:
                    # For non-retriable errors, just raise them
                    raise
            else:
                print(
                    f"Maximum retry attempts ({max_retries}) reached. Giving up.")
                raise

    # This should not be reached, but just in case
    raise Exception(f"Failed to make LLM API call after {max_retries} retries")


class ScenarioStats:
    """Class for tracking statistics related to scenario generation."""

    def __init__(self):
        """Initialize the statistics tracker."""
        self.total_scenarios = 0
        self.compiled_first_try = 0
        self.fixed_successfully = 0
        self.failed_scenarios = 0
        self.total_llm_calls = 0
        self.total_llm_calls_for_successful_fixes = 0
        self.fix_llm_calls = 0
        self.llm_calls_per_scenario = []
        self.min_llm_calls = float('inf')
        self.max_llm_calls = 0
        self.start_time = time.time()

    def add_gen_calls(self, num_llm_calls):
        """
        Add the number of LLM calls made during generation.

        Args:
            num_llm_calls: Number of GPT calls made
        """
        self.total_llm_calls += num_llm_calls

    def record_scenario(self, compiled_first_try, num_llm_calls, success):
        """
        Record statistics for a scenario.

        Args:
            compiled_first_try: Whether scenario compiled on the first try
            num_llm_calls: Number of LLM calls used for this scenario
            success: Whether the scenario was ultimately successful
        """
        self.total_scenarios += 1
        self.total_llm_calls += num_llm_calls
        self.fix_llm_calls += num_llm_calls
        self.llm_calls_per_scenario.append(num_llm_calls)

        if success:
            self.total_llm_calls_for_successful_fixes += num_llm_calls

        if compiled_first_try:
            self.compiled_first_try += 1
        elif success:
            self.fixed_successfully += 1
        else:
            self.failed_scenarios += 1

        if num_llm_calls < self.min_llm_calls:
            self.min_llm_calls = num_llm_calls
        if num_llm_calls > self.max_llm_calls:
            self.max_llm_calls = num_llm_calls

    def print_summary(self):
        """Print a summary of the statistics."""
        duration = time.time() - self.start_time

        print("\n" + "="*50)
        print("STATISTICS SUMMARY")
        print("="*50)
        print(f"Total scenarios processed: {self.total_scenarios}")
        print(
            f"Time elapsed: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        if self.total_scenarios > 0:
            print(
                f"Compiled on first try: {self.compiled_first_try} ({self.compiled_first_try/self.total_scenarios*100:.1f}%)")
            print(
                f"Fixed successfully: {self.fixed_successfully} ({self.fixed_successfully/self.total_scenarios*100:.1f}%)")
            print(
                f"Failed scenarios: {self.failed_scenarios} ({self.failed_scenarios/self.total_scenarios*100:.1f}%)")
            success_rate = (self.compiled_first_try +
                            self.fixed_successfully)/self.total_scenarios*100
            print(f"Total success rate: {success_rate:.1f}%")

            print("\nLLM CALL STATISTICS:")
            print(f"Total LLM calls: {self.total_llm_calls}")
            print(f"LLM calls for fixing: {self.fix_llm_calls}")
            avg_calls_for_fixes = self.total_llm_calls_for_successful_fixes / \
                self.fixed_successfully if self.fixed_successfully > 0 else 0
            print(
                f"Avg LLM calls for successful fixes: {avg_calls_for_fixes:.2f}")
            print(
                f"Average LLM calls per scenario: {self.total_llm_calls/self.total_scenarios:.2f}")
            if self.llm_calls_per_scenario:
                print(f"Min LLM calls for a scenario: {self.min_llm_calls}")
                print(f"Max LLM calls for a scenario: {self.max_llm_calls}")
        print("="*50)

    def save_to_json(self, file_path):
        """Save statistics to a JSON file."""
        avg_llm_calls_per_scenario = self.total_llm_calls / \
            self.total_scenarios if self.total_scenarios > 0 else 0
        avg_llm_calls_for_successful_fixes = self.total_llm_calls_for_successful_fixes / \
            self.fixed_successfully if self.fixed_successfully > 0 else 0
        success_rate_percent = (self.compiled_first_try + self.fixed_successfully) / \
            self.total_scenarios * 100 if self.total_scenarios > 0 else 0

        stats_dict = {
            "total_scenarios": self.total_scenarios,
            "compiled_first_try": self.compiled_first_try,
            "fixed_successfully": self.fixed_successfully,
            "failed_scenarios": self.failed_scenarios,
            "total_llm_calls": self.total_llm_calls,
            "fix_llm_calls": self.fix_llm_calls,
            "avg_llm_calls_per_scenario": avg_llm_calls_per_scenario,
            "avg_llm_calls_for_successful_fixes": avg_llm_calls_for_successful_fixes,
            "min_llm_calls": self.min_llm_calls if self.min_llm_calls != float('inf') else 0,
            "max_llm_calls": self.max_llm_calls,
            "success_rate_percent": success_rate_percent,
            "execution_time_seconds": time.time() - self.start_time,
            "llm_calls_per_scenario": self.llm_calls_per_scenario
        }
        with open(file_path, 'w') as f:
            json.dump(stats_dict, f, indent=4)

        print(f"Statistics saved to {file_path}")


def load(file_path):
    with open(file_path, 'r') as file:
        return file.read()


extraction_prompt = load(
    osp.join(local_path, 'ARISE', 'prompts', 'generation', 'extraction.txt'))
behavior_prompt = load(
    osp.join(local_path, 'ARISE', 'prompts', 'generation', 'behavior.txt'))
geometry_prompt = load(
    osp.join(local_path, 'ARISE', 'prompts', 'generation', 'geometry.txt'))
spawn_prompt = load(osp.join(local_path, 'ARISE',
                    'prompts', 'generation', 'spawn.txt'))
misc_prompt = load(osp.join(local_path, 'ARISE',
                   'prompts', 'generation', 'misc.txt'))
weather_prompt = load(
    osp.join(local_path, 'ARISE', 'prompts', 'generation', 'weather.txt'))
requirements_prompt = load(
    osp.join(local_path, 'ARISE', 'prompts', 'generation', 'requirements.txt'))

scenario_descriptions = load(
    osp.join(local_path, 'ARISE', 'scenario_descriptions.txt')).split('\n')

defaut_fix_prompt = load(
    osp.join(local_path, 'ARISE', 'prompts', 'fix', 'fix-default.txt'))
behavior_fix_prompt = load(
    osp.join(local_path, 'ARISE', 'prompts', 'fix', 'fix-behavior.txt'))
geometry_fix_prompt = load(
    osp.join(local_path, 'ARISE', 'prompts', 'fix', 'fix-geometry.txt'))
spawn_fix_prompt = load(
    osp.join(local_path, 'ARISE', 'prompts', 'fix', 'fix-spawn.txt'))
misc_fix_prompt = load(
    osp.join(local_path, 'ARISE', 'prompts', 'fix', 'fix-misc.txt'))
weather_fix_prompt = load(
    osp.join(local_path, 'ARISE', 'prompts', 'fix', 'fix-weather.txt'))
model = SentenceTransformer(
    'sentence-transformers/sentence-t5-large', device='cuda')

with open(osp.join(local_path, DB), 'rb') as file:
    database = pickle.load(file)

behavior_descriptions = database['behavior']['description']
geometry_descriptions = database['geometry']['description']
spawn_descriptions = database['spawn']['description']
misc_descriptions = database['misc']['description']
weather_descriptions = database['weather']['description']

behavior_snippets = database['behavior']['snippet']
geometry_snippets = database['geometry']['snippet']
spawn_snippets = database['spawn']['snippet']
misc_snippets = database['misc']['snippet']
weather_snippets = database['weather']['snippet']


behavior_embeddings = model.encode(
    behavior_descriptions, device='cuda', convert_to_tensor=True)
geometry_embeddings = model.encode(
    geometry_descriptions, device='cuda', convert_to_tensor=True)
spawn_embeddings = model.encode(
    spawn_descriptions, device='cuda', convert_to_tensor=True)
misc_embeddings = model.encode(
    misc_descriptions, device='cuda', convert_to_tensor=True)
weather_embeddings = model.encode(
    weather_descriptions, device='cuda', convert_to_tensor=True)

head = '''param map = localPath(f'../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"'''


def retrieve_topk(descriptions, snippets, embeddings, current_description, topk):
    current_embedding = model.encode(
        [current_description], device='cuda', convert_to_tensor=True)
    scores = (current_embedding @ embeddings.T).squeeze(0)
    top_indices = scores.topk(k=topk).indices
    top_descriptions = [descriptions[i] for i in top_indices]
    top_snippets = [snippets[i] for i in top_indices]

    return top_descriptions, top_snippets


def extract_scenic_code(text):
    pattern = r"```scenic(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    else:
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        return text.strip()


def test_scenario(scenario_path):
    # read the lines from the file and extract the map name, it is written as Town = 'xxx'
    with open(scenario_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Town ="):
                Town = line.split('=')[1].strip().strip("'")
                break

    scenario = scenic.scenarioFromFile(
        scenario_path, mode2D=True, model="scenic.simulators.carla.model")

    scene, _ = scenario.generate(maxIterations=1000)

    map_path = osp.join(
        local_path, 'scenic_scenarios', 'assets', 'maps', 'CARLA', f'{Town}.xodr')

    # Create simulator
    simulator = carla.CarlaSimulator(
        Town, map_path, '127.0.0.1', 2000, 10, False, False, 0.1)

    # Retry logic for CARLA world loading
    for attempt in range(MAX_CARLA_RETRIES):
        try:
            simulation = simulator.simulate(scene, maxSteps=1000)
            return True
        except RuntimeError as e:
            if "CARLA could not load world" in str(e):
                if attempt < MAX_CARLA_RETRIES - 1:  # Not the last attempt
                    print(
                        f"[WARNING] CARLA world loading failed (attempt {attempt + 1}/{MAX_CARLA_RETRIES}): {e}")
                    print(f"[INFO] Retrying in {CARLA_RETRY_DELAY} seconds...")
                    time.sleep(CARLA_RETRY_DELAY)
                    continue
                else:  # Last attempt, re-raise the error
                    print(
                        f"[ERROR] CARLA world loading failed after {MAX_CARLA_RETRIES} attempts: {e}")
                    raise
            else:
                # Different RuntimeError, re-raise immediately
                raise
        except Exception as e:
            # Non-RuntimeError exceptions, re-raise immediately
            raise

    # This should not be reached due to the logic above, but just in case
    return True


MAX_GENERATION_ATTEMPTS = 15
RETRY_DELAY_SECONDS = 0.5
MAX_CARLA_RETRIES = 6  # Maximum number of retries for CARLA world loading
CARLA_RETRY_DELAY = 10  # Delay in seconds between CARLA retries


def identify_code_section(line_number, line_ranges):
    """
    Maps a line number to its corresponding code section based on line ranges.

    Args:
        line_number (int): The line number from an exception
        line_ranges (dict): Dictionary mapping section names to (start, end) line tuples

    Returns:
        str: The name of the section containing this line, or "unknown" if not found
    """
    if not line_number or not isinstance(line_number, int):
        return "unknown"

    for section, range_tuple in line_ranges.items():
        # Skip None values or invalid ranges
        if not range_tuple:
            continue

        # Check that we have valid integers for comparison
        try:
            start, end = range_tuple
            if start <= line_number <= end:
                return section
        except (TypeError, ValueError):
            # Skip ranges that can't be properly compared
            continue

    return "unknown"


def _format_error_message(e: Exception, line_ranges=None):
    """
    Formats an error message, attempting to extract Scenic-specific details.
    Falls back to generic formatting with a traceback if specific attributes are missing.
    If line_ranges is provided, it will identify which code section an error occurred in.

    Args:
        e: The exception that occurred.
        line_ranges: Optional dictionary of code section line ranges

    Returns:
        A formatted string containing error details.
    """
    # Try to get Scenic-specific attributes (like from SimulationCreationError)
    if hasattr(e, 'filename') and hasattr(e, 'lineno') and hasattr(e, 'offset') and hasattr(e, 'msg') and hasattr(e, 'text'):
        filename = getattr(e, 'filename', 'Unknown file')
        lineno = getattr(e, 'lineno', 'Unknown line')
        offset = getattr(e, 'offset', 'Unknown character')
        msg = getattr(e, 'msg', 'No message')
        text = getattr(e, 'text', 'No text')

        section = "unknown"
        if line_ranges and isinstance(lineno, int):
            section = identify_code_section(lineno, line_ranges)

        return lineno, (f"File: {filename}, line: {lineno} (in {section} section), char: {offset}, "
                        f"Error: {msg} - '{text.strip()}'")
    else:
        # Generic exception
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)

        # Try to extract line number from traceback if possible
        section_info = ""
        if line_ranges:
            for line in tb_lines:
                if ", line " in line:
                    try:
                        lineno = int(line.split(", line ")[1].split(",")[0])
                        section = identify_code_section(lineno, line_ranges)
                        section_info = f" (in {section} section)"
                        break
                    except (ValueError, IndexError):
                        pass

        return None, (f"Error type: {type(e).__name__}, Message: {e}{section_info}\n"
                      f"Traceback:\n{''.join(tb_lines)}")


def _perform_compilation_attempts(file_path: str, total_attempts: int, context_description: str) -> tuple:
    """
    Handles compilation and retries for SimulationCreationError.

    Args:
        file_path: Path to the Scenic code file.
        total_attempts: Total number of simulation attempts to make.
        context_description: A string like "Initial" or "LLM fix attempt X".

    Returns:
        tuple: (success: bool, error_details: Optional[tuple(Exception, int, str)])
               error_details is (exception_instance, lineno, error_message) if success is False.
    """
    last_sce = None
    for attempt_num in range(1, total_attempts + 1):
        try:
            if attempt_num > 1:  # Delay for any retry
                time.sleep(RETRY_DELAY_SECONDS)
            test_scenario(file_path)

            # Success messages
            if context_description == "Initial":
                if attempt_num == 1:
                    print(
                        f"[SUCCESS] Code compiled successfully on the first attempt.")
                else:
                    print(
                        f"[SUCCESS] Code compiled successfully after {attempt_num} spawn attempts.")
            else:
                if attempt_num == 1:  # First try after LLM fix succeeded
                    print(
                        f"[SUCCESS] Code fixed and compiled successfully on {context_description}.")
                else:  # Succeeded on a spawn retry after LLM fix
                    print(
                        f"[SUCCESS] {context_description} worked after {attempt_num} additional spawn attempts.")
            return True, None
        except SimulationCreationError as sce:
            last_sce = sce
            if attempt_num == 1:  # First time this specific SimulationCreationError is encountered in this cycle
                if context_description == "Initial":
                    print(
                        f"[INFO] Initial compilation failed with SimulationCreationError (object spawning issue)")
                    if total_attempts > 1:
                        print(
                            f"[INFO] Starting up to {total_attempts - 1} more simulation attempts...")
                else:  # Post-LLM context
                    print(
                        f"[INFO] {context_description} resulted in SimulationCreationError. Trying additional spawn attempts up to {total_attempts}.")

            if attempt_num < total_attempts:
                continue  # Retry for SimulationCreationError
            else:  # Max attempts for SCE reached for this cycle
                print(
                    f"[ERROR] Object spawning failed after {total_attempts} attempts ({context_description}).")
                return False, (sce, *_format_error_message(sce))
        except Exception as e:
            # Other exceptions interrupt retries immediately
            formatted_lineno, formatted_error_message = _format_error_message(
                e)
            if context_description == "Initial":
                if attempt_num == 1:  # Failed on the very first try with a non-SCE error
                    print(
                        f"[ERROR] Compilation failed with {type(e).__name__}.")
                else:  # Failed with a non-SCE error during initial SCE retries
                    print(
                        f"[ERROR] Different error occurred on attempt {attempt_num} ({context_description}): {type(e).__name__}")
            else:  # Post-LLM context, non-SCE error
                print(
                    f"[ERROR] {context_description} didn't resolve the issue or introduced a new one: {type(e).__name__} - {str(e)}")
                # if error type name is AssertionError and str(e) == "", halt the process
                if type(e).__name__ == "AssertionError" and str(e) == "":
                    print(
                        f"[ERROR] AssertionError with empty message encountered. Halting further attempts.")
                    exit()
            return False, (e, formatted_lineno, formatted_error_message)

    # Fallback, should ideally be unreachable if total_attempts >= 1 and an exception always occurs on failure
    if last_sce:  # If loop finished due to exhausting attempts on SCE
        return False, (last_sce, *_format_error_message(last_sce))
    # This path should ideally not be hit if test_scenario always raises on error or returns normally.
    return False, (Exception("Unknown error: compilation attempts loop finished unexpectedly"), 0, "Unknown error")


def _handle_llm_correction(
    file_path: str,
    line_ranges,
    initial_error_details: tuple,
    max_llm_attempts: int,
    max_spawn_attempts_after_llm: int,
    llm_model,
    stats
) -> tuple:
    """
    Manages LLM fix attempts and subsequent recompilation.

    Args:
        initial_error_details: Tuple (exception_instance, lineno, error_message).
        max_spawn_attempts_after_llm: Max spawn attempts after an LLM fix (typically MAX_GENERATION_ATTEMPTS).

    Returns:
        tuple: (success: bool, llm_calls_made: int).
    """
    llm_calls_made = 0
    current_exception, current_lineno, current_error_message = initial_error_details

    print(
        f"[INFO] Starting LLM fix process with up to {max_llm_attempts} attempts")
    for llm_attempt_num in range(1, max_llm_attempts + 1):
        print(
            f"\n--- LLM Fix Attempt {llm_attempt_num}/{max_llm_attempts} ---")

        fix_applied_successfully, updated_line_ranges = _fix_with_llm(
            file_path, line_ranges, current_lineno, current_error_message, llm_model
        )
        llm_calls_made += 1
        if stats:
            # Assuming stats has a method like this, adjust if necessary
            # stats.increment_llm_calls()
            if hasattr(stats, 'llm_calls'):  # Simple example if stats is a dict or similar
                stats.llm_calls = getattr(stats, 'llm_calls', 0) + 1

        # Persist line_ranges changes for subsequent LLM calls
        line_ranges = updated_line_ranges

        compile_success = False

        if fix_applied_successfully:
            context = f"LLM fix attempt {llm_attempt_num}"
            compile_success, error_details_after_fix = _perform_compilation_attempts(
                file_path,
                max_spawn_attempts_after_llm,
                context
            )

            if compile_success:
                return True, llm_calls_made

            # Compilation after LLM fix failed, update error for next LLM attempt
            current_exception, current_lineno, current_error_message = error_details_after_fix
        else:
            print(
                f"[ERROR] LLM fix attempt {llm_attempt_num} failed to apply.")
            # Error info (current_exception, lineno, error_message) remains the same for the next LLM attempt.

        # Check compile_success for the case where fix applied but recompile failed
        if llm_attempt_num == max_llm_attempts and not compile_success:
            break

    # All LLM attempts are exhausted or fix failed to apply and then recompile failed
    print(
        f"[FAILURE] All {max_llm_attempts} LLM fix attempts were made and failed to produce compilable code.")
    return False, llm_calls_made


def test_scenic_code(file_path: str, line_ranges, max_llm_attempts, llm_model, stats=None) -> tuple:
    """
    Tests if Scenic code is compilable and attempts to fix it using LLM if necessary.

    The function follows this logic:
    1. Try compilation - if successful on first try, return True
    2. For SimulationCreationError (object spawning issues), retry up to MAX_GENERATION_ATTEMPTS times
    3. For other exceptions or if MAX_GENERATION_ATTEMPTS is exceeded, use LLM to fix
    4. If LLM fix is requested, make up to max_llm_attempts attempts

    Args:
        file_path: Path to the Scenic code file.
        line_ranges: Line ranges relevant for LLM fixes.
        max_llm_attempts: Maximum number of LLM correction attempts to make (default: 1)
        stats: Optional ScenarioStats instance to update with LLM call counts

    Returns:
        tuple: (success, llm_calls) where:
               - success is a boolean indicating if code compiled successfully
               - llm_calls is the number of LLM calls made (0 if none)
    """
    print(f"[INFO] Checking compilability of: {file_path}")
    llm_calls_total = 0

    initial_success, error_details = _perform_compilation_attempts(
        file_path,
        MAX_GENERATION_ATTEMPTS,  # Assumed global/module constant
        "Initial"
    )

    if initial_success:
        return True, llm_calls_total  # llm_calls_total is 0 here

    # Initial compilation failed. error_details should be populated.
    # Should have (exception, lineno, message)
    if not error_details or not error_details[0]:
        # This case should ideally not be reached if _perform_compilation_attempts is robust
        print(f"[ERROR] Initial compilation failed with an undetermined error.")
        print(f"[FAILURE] All fix attempts failed.")
        return False, llm_calls_total

    # If no LLM attempts are configured, we stop here.
    if max_llm_attempts <= 0:
        print(
            f"[INFO] Initial compilation failed. LLM attempts are disabled (--arise-fix-attempts={max_llm_attempts}).")
        print(f"[FAILURE] All fix attempts failed.")
        return False, llm_calls_total  # llm_calls_total is 0

    # STAGE 3: LLM fix attempts
    llm_success, llm_calls_this_stage = _handle_llm_correction(
        file_path,
        line_ranges,
        error_details,  # (exception_instance, lineno, error_message)
        max_llm_attempts,
        MAX_GENERATION_ATTEMPTS,  # Max spawn attempts after an LLM fix
        llm_model,
        stats
    )
    llm_calls_total += llm_calls_this_stage

    if llm_success:
        return True, llm_calls_total

    # If we reach here, all attempts (initial and LLM) have failed.
    # _handle_llm_correction would have printed its own summary if LLM attempts were made.
    # Final overall status message
    print(f"[FAILURE] All fix attempts failed.")
    return False, llm_calls_total


def _fix_with_llm(file_path: str, line_ranges, lineno, error_message: str, llm_model):
    """
    Helper function to fix Scenic code using LLM.

    Args:
        file_path: Path to the Scenic code file
        error_message: Formatted error message to send to LLM

    Returns:
        bool: True to indicate the fix was applied (not necessarily that it worked)
    """
    print(f"[INFO] Attempting code correction using LLM ({llm_model})...")

    try:
        # Read the current code
        with open(file_path, 'r') as file:
            scenic_code = file.read()

        # Prepare the prompt for the LLM API
        system_prompt = "You are an expert in the Scenic 3.0 programming language."

        # Identify the code section based on the line number, if any
        if lineno == None:
            code_section = "unknown"
        else:
            code_section = identify_code_section(lineno, line_ranges)

        print(f"\n\n{code_section} is being fixed\n\n")
        content = '\n'

        # Prepare the content for the prompt based on the code section
        if code_section == "unknown":  # default case, when no line number is specified
            user_prompt = defaut_fix_prompt.format(
                scenic_code=f"```scenic\n{scenic_code}```", error_message=f"```\n{error_message}\n```")
        elif code_section == "behavior":
            for j in range(topk):
                content += f'Description: {top_behavior_descriptions[j]}\nSnippet:\n```scenic\n{top_behavior_snippets[j]}```\n'
            user_prompt = behavior_fix_prompt.format(
                current_description=current_behavior, content=content, error_message=f"```\n{error_message}\n```", code=f"```scenic\n{scenic_code}```")
        elif code_section == "geometry":
            for j in range(topk):
                content += f'Description: {top_geometry_descriptions[j]}\nSnippet:\n```scenic\n{top_geometry_snippets[j]}```\n'
            user_prompt = geometry_fix_prompt.format(
                current_description=current_geometry, content=content, error_message=f"```\n{error_message}\n```", code=f"```scenic\n{scenic_code}```")
        elif code_section == "spawn":
            for j in range(topk):
                content += f'Description: {top_spawn_descriptions[j]}\nSnippet:\n```scenic\n{top_spawn_snippets[j]}```\n'
            user_prompt = spawn_fix_prompt.format(
                current_description=current_spawn, content=content, error_message=f"```\n{error_message}\n```", code=f"```scenic\n{scenic_code}```")
        elif code_section == "misc":
            for j in range(topk):
                content += f'Description: {top_misc_descriptions[j]}\nSnippet:\n```scenic\n{top_misc_snippets[j]}```\n'
            user_prompt = misc_fix_prompt.format(
                current_description=current_misc, content=content, error_message=f"```\n{error_message}\n```", code=f"```scenic\n{scenic_code}```")
        elif code_section == "weather":
            for j in range(topk):
                content += f'Description: {top_weather_descriptions[j]}\nSnippet:\n```scenic\n{top_weather_snippets[j]}```\n'
            user_prompt = weather_fix_prompt.format(
                current_description=current_weather, content=content, error_message=f"```\n{error_message}\n```", code=f"```scenic\n{scenic_code}```")
        elif code_section == "requirements":
            user_prompt = defaut_fix_prompt.format(
                scenic_code=f"```scenic\n{scenic_code}```", error_message=f"```\n{error_message}\n```")
        else:
            print(f"[ERROR] Unknown code section: {code_section}")
            return False, line_ranges

        response = make_llm_api_call(
            llm_model, system_prompt, user_prompt, client)

        # Extract the code from the response
        fixed_code = extract_scenic_code(response)
        if not fixed_code:
            print(
                "[WARN] Could not extract Scenic code from LLM response. Using the full response.")
            fixed_code = response  # Fallback

        # Save the corrected code
        with open(file_path, 'w') as file:
            file.write(fixed_code)
        line_ranges = update_line_ranges(fixed_code)
        print(
            f"[INFO] Potential fix from LLM applied and saved to {file_path}.")

        # Return true to indicate that the fix was applied (not necessarily that it worked)
        return True, line_ranges

    except Exception as api_or_file_error:
        print(
            f"[ERROR] Error retrieving/applying LLM fix: {api_or_file_error}")
        return False, line_ranges


def save_scenic_code(scenic_code, file_identifier, stats, iterations, llm_model):
    """
    Save Scenic code to a file and track statistics about compilation attempts.

    This function:
    1. Saves the scenic code to a file
    2. Tries to compile/test/fix the scenario using test_scenic_code
    3. Tracks all statistics including LLM calls

    Args:
        scenic_code: The Scenic code to save
        file_identifier: Unique identifier for the scenario file
        stats: ScenarioStats instance to track stats

    Returns:
        bool: Whether the code compiled successfully
    """
    # Create the file path with the provided file identifier
    file_path = osp.join(
        local_path, f'scenic_scenarios/ARISE_scenarios/test_scenic3_db/dynamic_{file_identifier}.scenic')

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as file:
        file.write(scenic_code)

    # Update line ranges based on the saved code
    line_ranges = update_line_ranges(scenic_code)

    initial_llm_calls = 0

    # Test the code and get success status plus additional LLM calls made
    success, llm_fix_calls = test_scenic_code(
        file_path, line_ranges, max_llm_attempts=iterations, llm_model=llm_model, stats=stats)

    # Total GPT calls = initial generation (1) + any LLM fix calls
    total_llm_calls = initial_llm_calls + llm_fix_calls

    # Record statistics with proper call counting
    compiled_first_try = (success and llm_fix_calls == 0)
    stats.record_scenario(compiled_first_try, total_llm_calls, success)

    # Log appropriate message
    if compiled_first_try:
        print(
            f"[SUCCESS] Scenic code compiled successfully on first attempt (1 LLM call)")
    elif success:
        print(
            f"[SUCCESS] Scenic code fixed and compiled successfully ({total_llm_calls} total LLM calls)")
    else:
        print(
            f"[FAILURE] Failed to compile the Scenic code after {total_llm_calls} total LLM calls")
        # change the file extension to .txt
        new_file_path = file_path.replace('.scenic', '.txt')
        try:
            os.rename(file_path, new_file_path)
        # Catching the exception if the file already exists
        except FileExistsError:
            os.remove(new_file_path)
            os.rename(file_path, new_file_path)
        print(f"[INFO] Saved failed scenario to {new_file_path}")

    return success, file_path


def update_line_ranges(scenic_code):
    """
    Updates line ranges based on section comments in the code.

    Args:
        scenic_code (str): The Scenic code with section comments

    Returns:
        dict: Updated dictionary mapping section names to (start, end) line tuples
    """
    lines = scenic_code.split('\n')
    line_ranges = {
        'header': None,
        'town': None,
        'imports': None,
        'weather': None,
        'behavior': None,
        'geometry': None,
        'spawn': None,
        'misc': None
    }

    # Find the header (first line)
    if lines and lines[0].startswith("'''"):
        line_ranges['header'] = (1, 1)

    current_section = None
    section_start = None

    # Go through the lines to find section comments and calculate ranges
    for i, line in enumerate(lines, 1):  # 1-based line numbering
        if line.startswith("# BEGIN "):
            section_name = line[8:].lower()  # Extract section name after BEGIN
            current_section = section_name
            section_start = i + 1  # Section content starts on the next line
        elif line.startswith("# END "):
            section_name = line[6:].lower()  # Extract section name after END
            if current_section == section_name and section_start is not None:
                # Add the range (excluding the BEGIN/END comment lines)
                line_ranges[current_section] = (section_start, i - 1)
                current_section = None
                section_start = None

    # print(f"Line ranges: {line_ranges}")

    return line_ranges


def extract_desc_parts(llm_response):
    """
    Extracts different parts of the scenario description from the LLM response.

    Args:
        llm_response (str): The response from the LLM containing scenario description parts.

    Returns:
        tuple: Extracted parts (object_type, behavior, geometry, spawn_position, misc, weather, requirements)
    """

    llm_response = llm_response.replace('*', '')

    match = re.search(
        r"Object Type:(.*?)Behavior:(.*?)Geometry:(.*?)Spawn Position:(.*)Miscellaneous:(.*)Weather:(.*)Requirements:(.*)", llm_response, re.DOTALL)

    if match:
        return [s.strip() for s in match.groups()]
    else:
        raise ValueError(
            "Could not extract scenario description parts from LLM response.")


def generate_scenarios(scenario_desc_path, llm_model, topk=2, count=1, iterations=10, max_attempts=5):
    """
        @param scenario_desc_path: Path to the scenario descriptions file
        @param model: LLM model to use for generation and fixes
        @param topk: Number of top snippets to retrieve for each component
        @param count: Number of scenarios to generate from each description
        @param iterations: Number of LLM fix attempts per scenario
        @param max_attempts: Maximum attempts to generate a valid scenario
        @return: path to the generated scenario on each iteration if successful, else None
    """

    try:
        if llm_model.startswith('gpt'):
            with open(osp.join(local_path, 'ARISE', 'openai_key.txt')) as file:
                key = file.read().strip()
                if key == '':
                    raise FileNotFoundError
                os.environ["OPENAI_API_KEY"] = key
            client = openai.OpenAI()
        elif llm_model.startswith('gemini') or llm_model.startswith('gemma'):
            with open(osp.join(local_path, 'ARISE', 'genai_key.txt')) as file:
                key = file.read().strip()
                if key == '':
                    raise FileNotFoundError
                os.environ["GOOGLE_API_KEY"] = key
            client = genai.Client(api_key=key)
        else:
            client = None

    except FileNotFoundError:
        print("Please provide the OpenAI API key in a file named 'openai_key.txt' or 'genai_key.txt' in the current directory.")
        exit(1)
    # read scenario descriptions from scenario_desc_path
    with open(scenario_desc_path, 'r') as f:
        scenario_descriptions = [line.strip()
                                 for line in f.readlines() if line.strip()]

    for q, curr_scenario in tqdm(enumerate(scenario_descriptions)):
        stats = ScenarioStats()
        system_prompt = "You are an expert in the Scenic 3.0 programming language."
        user_prompt = extraction_prompt.format(
            scenario=curr_scenario)
        # Generate each scenario count times
        for generation_num in range(count):
            attempt = 0
            success = False
            while attempt < max_attempts and not success:
                attempt += 1
                print(
                    f"\n=== Generating scenario {q + 1}, iteration {generation_num + 1}, attempt {attempt}/{max_attempts} ===\n")
                try:
                    resp = make_llm_api_call(
                        llm_model, system_prompt, user_prompt.format(scenario=curr_scenario), client)
                    current_adv_object, current_behavior, current_geometry, current_spawn, current_misc, current_weather, current_requirements = extract_desc_parts(
                        resp)

                    # retrieve the topk snippets for each component
                    # Retrieve snippets using embeddings
                    top_behavior_descriptions, top_behavior_snippets = retrieve_topk(
                        behavior_descriptions, behavior_snippets, behavior_embeddings, current_behavior, topk)
                    top_geometry_descriptions, top_geometry_snippets = retrieve_topk(
                        geometry_descriptions, geometry_snippets, geometry_embeddings, current_geometry, topk)
                    top_spawn_descriptions, top_spawn_snippets = retrieve_topk(
                        spawn_descriptions, spawn_snippets, spawn_embeddings, current_spawn, topk)
                    top_misc_descriptions, top_misc_snippets = retrieve_topk(
                        misc_descriptions, misc_snippets, misc_embeddings, current_misc, topk)
                    top_weather_descriptions, top_weather_snippets = retrieve_topk(
                        weather_descriptions, weather_snippets, weather_embeddings, current_weather, topk)

                    generated_behavior_code = make_llm_api_call(
                        llm_model,
                        system_prompt,
                        behavior_prompt.format(content='\n'.join([f'Description: {desc}\nSnippet:\n```scenic\n{snip}```\n' for desc, snip in zip(
                            top_behavior_descriptions, top_behavior_snippets)]), current_description=current_behavior),
                        client
                    )

                    generated_behavior_code = extract_scenic_code(
                        generated_behavior_code)

                    generated_geometry_code = make_llm_api_call(
                        llm_model,
                        system_prompt,
                        geometry_prompt.format(content='\n'.join([f'Description: {desc}\nSnippet:\n```scenic\n{snip}```\n' for desc, snip in zip(
                            top_geometry_descriptions, top_geometry_snippets)]), current_description=current_geometry),
                        client
                    )

                    generated_geometry_code = extract_scenic_code(
                        generated_geometry_code)

                    generated_spawn_code = make_llm_api_call(
                        llm_model,
                        system_prompt,
                        spawn_prompt.format(content='\n'.join([f'Description: {desc}\nSnippet:\n```scenic\n{snip}```\n' for desc, snip in zip(
                            top_spawn_descriptions, top_spawn_snippets)]), current_description=current_spawn),
                        client
                    )

                    generated_spawn_code = extract_scenic_code(
                        generated_spawn_code)

                    # Increment the LLM call count for generation
                    stats.add_gen_calls(4)

                    if current_misc.strip() != "None":
                        misc_prompt_content = '\n'.join([f'Description: {desc}\nSnippet:\n```scenic\n{snip}```\n' for desc, snip in zip(
                            top_misc_descriptions, top_misc_snippets)])
                        response = make_llm_api_call(
                            llm_model,
                            system_prompt,
                            misc_prompt.format(
                                content=misc_prompt_content, current_description=current_misc, spawn_desc=current_spawn),
                            client
                        )
                        generated_misc_code = extract_scenic_code(response)
                        stats.add_gen_calls(1)
                    else:
                        generated_misc_code = ""

                    if current_weather.strip() != "Any":
                        weather_prompt_content = '\n'.join([f'Description: {desc}\nSnippet:\n```scenic\n{snip}```\n' for desc, snip in zip(
                            top_weather_descriptions, top_weather_snippets)])
                        response = make_llm_api_call(
                            llm_model,
                            system_prompt,
                            weather_prompt.format(
                                content=weather_prompt_content, current_description=current_weather),
                            client
                        )
                        generated_weather_code = extract_scenic_code(response)
                        stats.add_gen_calls(1)
                    else:
                        generated_weather_code = ""

                    if current_requirements.strip() != "":
                        generated_requirements_code = make_llm_api_call(
                            llm_model,
                            system_prompt,
                            requirements_prompt.format(
                                current_description=curr_scenario, scenic_code=("```scenic\n" + generated_geometry_code + "\n" + generated_spawn_code + "\n" + generated_misc_code + "```"), requirements=current_requirements),
                            client
                        )
                        generated_requirements_code = extract_scenic_code(
                            generated_requirements_code)
                        stats.add_gen_calls(1)
                    else:
                        generated_requirements_code = ""

                    Town, generated_geometry_code = generated_geometry_code.split(
                        '\n', 1)

                    try:
                        header = f"'''{curr_scenario}'''"

                        # Assemble the code with section comments
                        parts = []
                        parts.append(header)  # Scenario description
                        parts.append("# BEGIN TOWN")
                        parts.append(Town)
                        parts.append("# END TOWN")
                        parts.append("# BEGIN IMPORTS")
                        parts.append(head)
                        parts.append("# END IMPORTS")

                        if generated_weather_code != "":
                            parts.append("# BEGIN WEATHER")
                            parts.append(generated_weather_code)
                            parts.append("# END WEATHER")

                        parts.append("# BEGIN BEHAVIOR")
                        parts.append(generated_behavior_code)
                        parts.append("# END BEHAVIOR")

                        parts.append("# BEGIN GEOMETRY")
                        parts.append(generated_geometry_code)
                        parts.append("# END GEOMETRY")

                        # Format spawn code with AdvObject if it contains the format placeholder
                        spawn_code = generated_spawn_code.format(
                            AdvObject=current_adv_object)
                        parts.append("# BEGIN SPAWN")
                        parts.append(spawn_code)
                        parts.append("# END SPAWN")

                        if generated_misc_code != "":
                            parts.append("# BEGIN MISC")
                            parts.append(generated_misc_code)
                            parts.append("# END MISC")

                        parts.append("# BEGIN REQUIREMENTS")
                        parts.append(generated_requirements_code)
                        parts.append("# END REQUIREMENTS")

                        scenic_code = '\n'.join(parts)
                    except KeyError as e:
                        # Fallback method if formatting fails
                        print(
                            f"Warning: Could not format spawn code with AdvObject: {e}")

                        # Use the same approach but without formatting the spawn code
                        try:
                            parts.append(generated_spawn_code)

                            if generated_misc_code != "":
                                parts.append(generated_misc_code)

                            scenic_code = '\n'.join(parts)
                        except Exception as line_count_error:
                            print(
                                f"Warning: Error calculating line numbers: {line_count_error}")
                            # fallback without line tracking
                            scenic_code = '\n'.join([
                                f"'''{curr_scenario}'''",
                                Town,
                                head,
                                generated_weather_code,
                                generated_behavior_code,
                                generated_geometry_code,
                                generated_spawn_code,
                                generated_misc_code
                            ])

                    # Use a unique identifier for each generation of the same scenario
                    file_identifier = f"{q}_{generation_num}"

                except Exception as e:
                    print(
                        f"[ERROR] Error during scenario generation: {type(e).__name__} - {str(e)}")
                    continue
                # Save and test the scenic code
                success, file_path = save_scenic_code(
                    scenic_code, file_identifier, stats, iterations, llm_model)
                generated_scenarios.append((file_path, success))

            if not success:
                print(
                    f"[FAILURE] Could not generate a valid scenario after {max_attempts} attempts for scenario {q + 1}, iteration {generation_num + 1}.")
                generated_scenarios.append((None, success))

    return generated_scenarios
