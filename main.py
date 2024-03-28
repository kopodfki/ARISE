import os
import argparse
from argparse import RawTextHelpFormatter
from ARISE.logger_config import logger, set_debug_logging
import sys


def parse_arguments():
    """
    Parses command-line arguments.
    """

    parser = argparse.ArgumentParser(
        prog="ARISE",
        description="LLM-driven scenic scenario generator.",
        formatter_class=RawTextHelpFormatter,
    )

    # Logging and generation arguments
    parser.add_argument("-v", "--debug", default=False,
                        action="store_true", help="Prints debug information.")

    # Configuration arguments
    parser.add_argument(
        "--arise-description", type=str, default=None, help="Provide path to a file containing scenario descriptions for LLM-based scenario generation."
    )
    parser.add_argument(
        "--arise-model", type=str, default="gpt-5.1", help="Specify the LLM model to use for scenario generation. (default: gpt-5.1)"
    )
    parser.add_argument(
        "--arise-topk", type=int, default=2, help="Number of snippets to use during LLM generation. (default: 2)"
    )
    parser.add_argument(
        "--arise-count", type=int, default=1, help="Number of scenarios to generate using LLM from each description. (default: 1)"
    )
    parser.add_argument(
        "--arise-fix-attempts", type=int, default=10, help="Number of LLM fix attempts per scenario. (default: 10)"
    )
    parser.add_argument(
        "--arise-max-attempts", type=int, default=5, help="Maximum attempts to generate a valid scenario using LLM. (default: 5)"
    )
    args = parser.parse_args()

    # Validate scenario and map arguments
    if args.arise_description is None:
        raise ValueError(
            "Provide --arise-description.")
    else:
        # validate that the arise_description file exists
        if not os.path.isfile(args.arise_description):
            raise ValueError(
                f"The file specified in --arise-description does not exist: {args.arise_description}"
            )

    return args


def main():
    try:
        args = parse_arguments()
    except ValueError as e:
        logger.critical(e)
        sys.exit(1)

    if args.debug:
        set_debug_logging()

    # os.system("color")  # For colored output on Windows
    try:
        # LLM-based scenario generation
        logger.info(f"Generating scenario using LLM ({args.arise_model}).")
        from ARISE.retrieve import generate_scenarios
        description = args.arise_description
        model = args.arise_model
        topk = args.arise_topk
        count = args.arise_count
        fix_attempts = args.arise_fix_attempts
        max_attempts = args.arise_max_attempts
        # Generates scenarios and returns a list of tuples (file_path, success_flag)
        arise_gen_scenarios = generate_scenarios(
            description, model, topk, count, fix_attempts, max_attempts)

        # Print summary of generated scenarios
        success_count = sum(1 for _, success in arise_gen_scenarios if success)
        logger.info(
            f"Scenario generation completed: {success_count}/{len(arise_gen_scenarios)} scenarios generated successfully.")
    except KeyboardInterrupt:
        logger.info("Cancelled by user. Exiting...")


if __name__ == "__main__":
    main()
