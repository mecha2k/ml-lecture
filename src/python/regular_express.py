import re


def main():
    with open("data/in.txt", "r") as f:
        line = f.readline()
        x = re.findall('Hi|Hello', line)
        print(x)
        # lines = f.readlines()
        # for line in lines:
        #     x = re.search("^Hi$", line)
        #     print(line.strip())
        #     print(x)





if __name__ == "__main__":
    main()
