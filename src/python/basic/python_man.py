import numpy as np


def iterate_batches(batch_size=16):
    while True:
        batch = np.random.randint(0, 100, batch_size)
        yield batch


def createGen():
    myList = range(5)
    for i in myList:
        yield i + 5


def main():
    a = iterate_batches()
    print(a)
    for b, c in enumerate(a):
        print(b, c)
        if b == 10:
            break

    mygenerator = (x**2 for x in range(5))
    for x in mygenerator:
        print(x)
    for x1 in mygenerator:
        print(x1 + 1)

    gen = createGen()
    print(gen)
    for x in gen:
        print(x)


if __name__ == "__main__":
    main()
