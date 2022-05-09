import os
import random
import pymysql
from dotenv import load_dotenv

load_dotenv(verbose=True)

fnames = list("김이박최강고윤엄한배성백전황서천방지마피" * 2)
lnames = list("건성현욱정민현주희진영래주동해도모양지선재현호시우인성마무병별솔하라기" * 2)
phones = list("0123456789" * 3)
emails = list("abcdefghijklmnopqrstuvwxyz" * 3)
addres = ["서울", "부산", "대구", "광주", "대전", "원주", "울산", "포항", "강릉", "천안", "인천", "청주"]


def make_sample():
    fname = random.sample(fnames, k=1)
    lname = "".join(random.sample(lnames, k=2))
    name = str(fname[0]) + lname

    phone1 = "".join(random.sample(phones, k=4))
    phone2 = "".join(random.sample(phones, k=4))
    phone = f"010-{phone1}-{phone2}"

    email = "".join(random.sample(emails, k=random.randrange(5, 10)))
    email = f"{email}@naver.com"

    address = random.sample(addres, k=1)[0]

    years = list(range(1960, 2000))
    months = list(range(1, 13))
    days31 = list(range(1, 32))
    days30 = list(range(1, 31))
    days28 = list(range(1, 29))
    m30 = [4, 6, 9, 11]

    yy = random.choice(years)
    mm = random.choice(months)
    dd = random.choice(days31)
    if mm in m30 and dd > 30:
        dd = random.choice(days30)
    elif mm == 2 and dd > 28:
        dd = random.choice(days28)
    birth = f"{yy:4d}-{mm:02d}-{dd:02d}"

    return name, address, birth, phone, email


if __name__ == "__main__":
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWD"),
        db=os.getenv("DB_NAME"),
        charset="utf8",
        port=3306,
    )
    print(conn)
    print(make_sample())

    data = []
    for i in range(1000):
        data.append(make_sample())

    with conn:
        cursor = conn.cursor()
        sql = "INSERT INTO Student(name, address, birth, phone, email) VALUES(%s, %s, %s, %s, %s)"
        cursor.executemany(sql, data)
        print("Affected RowCount is", cursor.rowcount)
        conn.commit()
