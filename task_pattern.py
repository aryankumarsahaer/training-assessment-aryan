# task_pattern.py

n = int(input("Enter n: "))

if n < 5 or n % 2 == 0:
    print("Please enter an odd integer >= 5")

else:
    mid = n // 2

    for i in range(n):

        # Middle row
        if i == mid:
            print("*" * (n + mid + 1))

        # Upper half
        elif i < mid:
            print("*" + " " * n + "*" * (i + 1))

        # Lower half
        else:
            print("*" + " " * n + "*" * (n - i))