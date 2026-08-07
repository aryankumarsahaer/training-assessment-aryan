#!/usr/bin/env python3
"""
Task 1: Pattern Generation
Write a Python script to generate a dynamic arrow star pattern for any odd integer input n >= 5.
"""

import sys
import argparse

def generate_arrow_pattern(n: int) -> str:
    """
    Generates an arrow star pattern.
    For input n = 5:
    *****
     ****
      ***
       **
        *
       **
      ***
     ****
    *****
    """
    lines = []
    
    # Upper part (including the middle line of 1 star)
    for i in range(n):
        spaces = " " * i
        stars = "*" * (n - i)
        lines.append(spaces + stars)
        
    # Lower part (mirror of upper part, starting from 2 stars)
    for i in range(n - 2, -1, -1):
        spaces = " " * i
        stars = "*" * (n - i)
        lines.append(spaces + stars)
        
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a dynamic arrow star pattern for any odd integer input n >= 5."
    )
    parser.add_argument(
        "-n", "--size",
        type=int,
        help="An odd integer size >= 5"
    )
    args = parser.parse_args()

    # If no argument is provided, prompt the user interactively
    if args.size is None:
        try:
            val_str = input("Enter an odd integer (n >= 5): ").strip()
            if not val_str:
                print("Error: No input provided.")
                sys.exit(1)
            n = int(val_str)
        except ValueError:
            print("Error: Input must be a valid integer.")
            sys.exit(1)
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(0)
    else:
        n = args.size

    # Validate constraints
    if n < 5:
        print(f"Error: Size must be at least 5 (got {n}).")
        sys.exit(1)
    if n % 2 == 0:
        print(f"Error: Size must be an odd integer (got {n}, which is even).")
        sys.exit(1)

    # Generate and print pattern
    pattern = generate_arrow_pattern(n)
    print(f"\nArrow Star Pattern (n = {n}):")
    print("-" * (2 * n - 1))
    print(pattern)
    print("-" * (2 * n - 1))

if __name__ == "__main__":
    main()
