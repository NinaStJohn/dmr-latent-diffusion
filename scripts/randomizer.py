import random

def shuffle_file_lines(filename):
    # Read all lines
    with open(filename, "r") as f:
        lines = f.readlines()

    # Shuffle the lines
    random.shuffle(lines)

    # Write back to the same file
    with open(filename, "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    fname = input("Enter the filename: ")
    shuffle_file_lines(fname)
    print(f"Lines shuffled and written back to {fname}")
