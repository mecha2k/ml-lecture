SHOW PROCESSLIST;
SHOW DATABASES;
USE mycnlife;

CREATE TABLE `Student` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `address` varchar(45) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `birth` date DEFAULT NULL,
  `phone` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `email` varchar(30) COLLATE utf8mb4_unicode_ci NOT NULL,
  `regdate` timestamp NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

SHOW TABLE STATUS;
SHOW TABLES;

ALTER TABLE `mycnlife`.`Student` CHANGE COLUMN 
`regdate` `regdate` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP;
DESCRIBE Student;

INSERT INTO Student(name, address, birth, phone, email) 
VALUES('김일수', '서울', '2003-01-02', '010-1111-2222', 'onekim@gmail.com');
INSERT INTO Student(name, address, birth, phone, email) 
VALUES('김이수', '부산', '2005-03-22', '010-2222-2222', 'twokim@gmail.com');
INSERT INTO Student(name, address, birth, phone, email) 
VALUES('김삼수', '광주', '2007-11-02', '010-3333-2222', 'threekim@gmail.com');

DESC Student;
SELECT * FROM Student;
SHOW INDEX FROM Student;
SELECT * FROM Student ORDER BY id ASC;
SELECT * FROM Student WHERE name LIKE '%정현';
SELECT * FROM Student WHERE id IN (10, 20, 30);
SELECT * FROM Student WHERE id = 10 OR id = 20 OR id = 30;
SELECT * FROM Student WHERE id NOT BETWEEN 10 AND 30;
SELECT * FROM Student WHERE email LIKE 'a%' and phone LIKE '010-9%';

ALTER TABLE Student ADD COLUMN altcol int;
ALTER TABLE Student DROP COLUMN altcol;
UPDATE Student SET phone='010-3333-3333' WHERE id=5;
UPDATE Student SET name='김이수' WHERE id=3;
DELETE FROM Student WHERE id=2;
DELETE FROM Student WHERE id=4;
DELETE altcol FROM Student;
SELECT birth, replace(substring(birth, 3), '-', '') FROM Student;
SELECT * FROM Student WHERE birth IS NOT NULL;
TRUNCATE Student;

CREATE TABLE TestStu LIKE Student;
INSERT INTO TestStu SELECT * FROM Student;
SELECT * FROM TestStu;
TRUNCATE TestStu;
DROP TABLE TestStu;


