def arg_func(*args):
    print(len(args))
    for arg in args:
        print(arg)


def kwarg_func(**kwargs):
    print(len(kwargs))
    for key, value in kwargs.items():
        print(f"key : {key}, value : {value}")


class classDemo:
    def __init__(self, name):
        self.name = name
        self.data = [1, 2, 3]

    def my_func(self):
        return f"my func is {self.name}"

    def __repr__(self):
        return f"name is {self.name}"

    def __call__(self, value):
        print(f"__call__ : {value}")

    def __add__(self, other):
        return f"me is {self.name}, other is {other.name}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, position):
        return f"{self.name}__{self.data[position]}"


def decorate_func(func):
    def wrapper(*args, **kwargs):
        print("my pre-process")
        print(func(*args, **kwargs))
        print("my post-process")

    return wrapper


@decorate_func
def decorate_demo():
    print("It is decorator demo...")


def decorate_args(arg):
    print(f"decorate_args : {arg}")

    def decorate_func(func):
        def wrapper(*args, **kwargs):
            print(f"wrapper: {arg}")
            decovalue = func(*args, **kwargs)
            print(decovalue * 5)

        return wrapper

    return decorate_func


@decorate_args("decorate arg")
def decorate_demo1(arg):
    return arg


def main():
    arg_func()
    arg_func("apple", "orange")
    kwarg_func()
    kwarg_func(key1="apple", key2="banana", key3="orange")

    a = classDemo("home")
    b = classDemo("school")

    print(a + b)
    print(a.my_func())

    a("good")
    print(len(a))

    for item in a:
        print(item)
    print(a[1])

    decorate_demo()
    decorate_demo1(3)


if __name__ == "__main__":
    main()
