# task_ascii.py

def reduce_to_single_digit(number):
    # Base case
    if number < 10:
        return number

    # Convert number into individual digits
    digits = [int(digit) for digit in str(number)]

    # Calculate sum
    digit_sum = sum(digits)

    # Show calculation
    calculation = " + ".join(str(digit) for digit in digits)
    print(f"{calculation} = {digit_sum}")

    # Recursion
    return reduce_to_single_digit(digit_sum)

# Take name from user
name = input("Enter your name: ").upper()

total = 0

print("\nASCII Values:")

# Convert characters to ASCII
for char in name:
    ascii_value = ord(char)
    print(f"{char} -> {ascii_value}")
    total += ascii_value


print(f"\nASCII Sum = {total}")

print("\nDigit Sum Process:")

# Reduce to single digit
result = reduce_to_single_digit(total)

print(f"\nFinal Single Digit = {result}")