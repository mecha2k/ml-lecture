import re


def main():
    with open("data/in.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            x = re.findall("Hi|Hello", line, re.IGNORECASE)
            y = re.findall("grey", line)
            z = re.search("grey", line)
            print(line.strip())
            print(x)
            print(y)
            if z is not None:
                print(z.group())

            print(re.findall("gr[a-z]y", line, re.IGNORECASE))
            print(re.findall("gr..y", line))
            print(re.findall("^Hi|Hello$", line, re.IGNORECASE | re.MULTILINE))
            print(re.findall("[a-zA-Z0-9]", line))
            print(re.findall("[^a-zA-Z0-9]", line))
            print(re.findall("gr[aed]y", line))
            print(re.findall("gra*y", line))
            print(re.findall("gra+y", line))
            print(re.findall("gra{2,3}y", line))
            print(re.findall("\bYa", line, re.MULTILINE))
            print(re.findall("\d{2,3}[. -]\d{3}[. -]\d{4}", line))
            print(re.findall("[a-zA-Z0-9._+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.]+", line))
            print(re.findall("(https?://)?(www\.)?(youtu.be/)([a-zA-Z0-9-]+)", line))
            print(re.findall("(?:https?://)?(?:www\.)?youtu.be/[a-zA-Z0-9-]+", line))
            x = re.findall("((?:https?://)?(?:www\.)?youtu.be/)([a-zA-Z0-9-]+)", line)
            print("groups : ", x)


if __name__ == "__main__":
    main()
