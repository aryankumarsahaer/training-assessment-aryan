#!/usr/bin/env python3
"""
Task 3: ASCII Reduction
Convert a name to ASCII values, sum them, and recursively reduce to a single digit.
"""

import sys
import argparse

def reduce_name_to_digit(name: str, verbose: bool = True) -> tuple[int, list[str]]:
    """
    Converts a name to ASCII values, sums them, and recursively reduces the sum to a single digit.
    Returns a tuple of (result_digit, list_of_log_lines).
    """
    logs = []
    
    if not name:
        return 0, ["Error: Name cannot be empty."]

    # Step 1: Character to ASCII conversion
    char_ascii = []
    ascii_values = []
    for char in name:
        val = ord(char)
        char_ascii.append(f"'{char}' -> {val}")
        ascii_values.append(val)
    
    total_sum = sum(ascii_values)
    
    logs.append(f"Input Name: \"{name}\"")
    logs.append("Character ASCII mappings:")
    for mapping in char_ascii:
        logs.append(f"  {mapping}")
    
    logs.append(f"Initial Sum: {' + '.join(str(v) for v in ascii_values)} = {total_sum}")
    
    # Step 2: Recursive digit reduction
    current_val = total_sum
    step_num = 1
    
    while current_val >= 10:
        digits = [int(d) for d in str(current_val)]
        next_val = sum(digits)
        logs.append(f"Reduction Step {step_num}: Summing digits of {current_val} -> {' + '.join(str(d) for d in digits)} = {next_val}")
        current_val = next_val
        step_num += 1
        
    logs.append(f"Final Single Digit: {current_val}")
    
    if verbose:
        for line in logs:
            print(line)
            
    return current_val, logs

def main():
    parser = argparse.ArgumentParser(
        description="Convert a name to ASCII values, sum them, and recursively reduce to a single digit."
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        help="The name string to convert and reduce"
    )
    args = parser.parse_args()

    # If no name is provided, prompt interactively
    if args.name is None:
        try:
            name = input("Enter a name: ").strip()
            if not name:
                print("Error: Name cannot be empty.")
                sys.exit(1)
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(0)
    else:
        name = args.name

    print("\n=== ASCII Reduction ===")
    reduce_name_to_digit(name, verbose=True)
    print("=======================\n")

if __name__ == "__main__":
    main()
