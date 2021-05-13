import os, random
import pymysql
from dotenv import load_dotenv

load_dotenv(verbose=True)

fnames = list("김이박최강고윤엄한배성백전황서천방지마피" * 2)
lnames = list("건성현욱정민현주희진영래주동해도모양지선재현호시우인성마무병별솔하라기" * 2)
phones = list("0123456789" * 3)
emails = list("abcdefghijklmnopqrstuvwxyz" * 3)
addres = ["서울", "부산", "대구", "광주", "대전", "원주", "울산", "포항", "강릉", "천안", "인천", "청주"]


def make_sample():
    text = []
    for nn in range(5):
        phrase = "".join(random.sample(emails, k=random.randrange(3, 10)))
        text.append(phrase)
    title = " ".join(text)
    print(title)

    return title, random.randrange(0, 9)


if __name__ == "__main__":
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWD"),
        db=os.getenv("DB_NAME"),
        charset="utf8",
        port=3306,
    )

    data = []
    for i in range(200):
        data.append(make_sample())
    print(data)

    with conn:
        cursor = conn.cursor()
        sql = "INSERT INTO survey(title, state) VALUES(%s, %s)"
        cursor.executemany(sql, data)
        print("Affected RowCount is", cursor.rowcount)
        conn.commit()
