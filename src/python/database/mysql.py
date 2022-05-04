import pymysql
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
passwd = os.getenv("dbPASS")

conn = pymysql.connect(
    host="localhost", user="mecha2k", password=passwd, database="trading", charset="utf8mb4"
)

print(conn.user)
print(conn.unix_socket)

# with connection:
#     with connection.cursor() as cursor:
#         # Create a new record
#         sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
#         cursor.execute(sql, ("webmaster@python.org", "very-secret"))
#
#     # connection is not autocommit by default. So you must commit to save
#     # your changes.
#     connection.commit()
#
#     with connection.cursor() as cursor:
#         # Read a single record
#         sql = "SELECT `id`, `password` FROM `users` WHERE `email`=%s"
#         cursor.execute(sql, ("webmaster@python.org",))
#         result = cursor.fetchone()
#         print(result)
