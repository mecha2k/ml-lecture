SHOW DATABASES;

USE mysql;
-- node.js mysql8.0 connection problem solution
ALTER USER 'root'@'localhost' IDENTIFIED BY 'mySql80!'; 
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'mySql80!';
ALTER USER 'mecha2k'@'localhost' IDENTIFIED BY 'Techno88!'; 
ALTER USER 'mecha2k'@'localhost' IDENTIFIED WITH mysql_native_password BY 'Techno88!';

SELECT sha2('nodeJS', 256), sha2(concat('nodeJS', 'nodevue_mecha2Clubk'), 256);
SELECT * FROM Club where id = 2;
SELECT * FROM Department;

SHOW PROCESSLIST;
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
SELECT * FROM Student;

SELECT * FROM Student ORDER BY name ASC LIMIT 10, 30;

-- 서울지역 학생 중 어린순서로 11번째부터 5명 추출
SELECT * FROM Student WHERE address = '서울' ORDER BY birth DESC LIMIT 10, 5;

-- 지역별 학생수
SELECT address, COUNT(*) FROM Student group by address;
select address, count(*) as 'count' from Student GROUP BY address order by 'count' desc;

-- 지역별 학생수가 80명 이상이 지역들만 추출
select address, count(*) as cnt from Student GROUP BY address having cnt > 80 order by cnt desc;

CREATE TABLE `Club` (
  `id` smallint unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(30) COLLATE utf8mb4_unicode_ci NOT NULL,
  `createdate` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `leader` int unsigned DEFAULT NULL,
  PRIMARY KEY (`id`),
  foreign key (leader) references Student(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
DROP TABLE Club;
DESC Club;
SHOW CREATE TABLE Club;
DESC Student;

CREATE TABLE `mycnlife`.`Profess` (
  `id` SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(30) NOT NULL,
  `likecnt` INT NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`));
DROP TABLE Profess;

CREATE TABLE `mycnlife`.`Subject` (
  `id` SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(45) NOT NULL,
  `profess` SMALLINT NULL,
  PRIMARY KEY (`id`));
DROP TABLE Subject;

ALTER TABLE `mycnlife`.`Subject` 
CHANGE COLUMN `profess` `profess` SMALLINT UNSIGNED NULL DEFAULT NULL ;
ALTER TABLE `mycnlife`.`Subject` 
ADD CONSTRAINT `fk_subject_profess`
  FOREIGN KEY (`id`)
  REFERENCES `mycnlife`.`Profess` (`id`)
  ON DELETE RESTRICT
  ON UPDATE RESTRICT;
ALTER TABLE `mycnlife`.`Subject` DROP FOREIGN KEY `fk_subject_profess`;

ALTER TABLE `mycnlife`.`Subject` ADD CONSTRAINT FOREIGN KEY (`profess`) REFERENCES `mycnlife`.`Profess` (`id`)
  ON DELETE RESTRICT ON UPDATE RESTRICT;

CREATE TABLE `mycnlife`.`Enroll` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
  `subject` SMALLINT UNSIGNED NOT NULL,
  `student` INT UNSIGNED NOT NULL,
  PRIMARY KEY (`id`));
DROP TABLE Enroll;

ALTER TABLE Enroll ADD CONSTRAINT FOREIGN KEY (subject) REFERENCES Subject (id) ON DELETE CASCADE;
ALTER TABLE Enroll ADD CONSTRAINT FOREIGN KEY (student) REFERENCES Student (id) ON DELETE CASCADE;

DESC Enroll;
SELECT * FROM Club;

INSERT INTO Club (name, leader) VALUES ('요트부', 100);
INSERT INTO Club (name, leader) VALUES ('음악부', 200);
INSERT INTO Club (name, leader) VALUES ('미술부', 300);
INSERT INTO Club (name) VALUES ('유도부');

SELECT *, Student.name as 'student name' FROM Club INNER JOIN Student on Club.leader = Student.id;
SELECT Club.*, Student.name as 'student name' FROM Club INNER JOIN Student on Club.leader = Student.id;

DESCRIBE Profess;
SELECT ceil(rand()*10) FROM DUAL;
INSERT INTO Profess(name, likecnt) SELECT name, ceil(rand() * 100) FROM Student ORDER BY rand() LIMIT 100;
SELECT * FROM Profess;

DESCRIBE Subject;
INSERT INTO Subject(name, profess) SELECT '국어', id FROM Profess ORDER BY rand() LIMIT 10;
SELECT * FROM Subject;
UPDATE Subject SET name='체육' WHERE name='국어' and id <> 10 LIMIT 1;
ALTER TABLE Subject ADD UNIQUE INDEX (name ASC);

DESCRIBE Enroll;
SELECT * FROM Enroll;
INSERT INTO Enroll(student, subject) SELECT id, (SELECT id FROM Subject ORDER BY rand() LIMIT 1) 
FROM Student ORDER BY id;
INSERT INTO Enroll(student, subject) SELECT id, (SELECT id FROM Subject ORDER BY rand() LIMIT 1) 
FROM Student ORDER BY rand() LIMIT 500;
INSERT INTO Enroll(student, subject) SELECT id, (SELECT id FROM Subject ORDER BY rand() LIMIT 1) 
FROM Student ORDER BY rand() LIMIT 500 ON DUPLICATE KEY UPDATE student = student;
SELECT student, count(*) FROM Enroll GROUP BY student;
SELECT count(distinct student) FROM Enroll;

-- 과목별 담당 교수명
SELECT Subject.*, Profess.name FROM Subject INNER JOIN Profess ON Profess.id = Subject.profess;

-- 과목별 학생수
DESC Enroll;
SELECT Enroll.subject, max(Subject.name), count(*) FROM Enroll 
INNER JOIN Subject on Subject.id = Enroll.subject GROUP BY Enroll.subject;

-- 역사 과목의 학생 목록
SELECT Subject.name, Enroll.student, Student.name FROM Enroll 
INNER JOIN Subject ON Subject.id = Enroll.subject 
INNER JOIN Student ON Student.id = Enroll.student
WHERE Subject.name = '역사'; 

SELECT Subject.name, count(*) FROM Enroll 
INNER JOIN Subject ON Subject.id = Enroll.subject WHERE Subject.name = '역사' GROUP BY Enroll.subject;

-- 특정과목(국어)과목을 듣는 서울 거주 학생 목록 (과목명, 학번, 학생명)
SELECT Subject.name, Student.id, Student.name FROM Enroll 
INNER JOIN Student ON Student.id = Enroll.student
INNER JOIN Subject ON Subject.id = Enroll.subject
WHERE Subject.name = '국어' and Student.address = '서울';
SELECT * FROM Enroll WHERE subject = 10;

-- 역사 과목을 수강중인 지역별 학생수 (지역, 학생수)
SELECT Student.address, count(*) FROM Enroll
INNER JOIN Student ON Student.id = Enroll.student
INNER JOIN Subject ON Subject.id = Enroll.subject
WHERE Subject.name = '역사' GROUP BY Student.address;

SELECT * FROM enroll_view_01;
SELECT * FROM Club;
INSERT INTO Club(name, leader) VALUE('검도부', NULL);
SELECT Club.*, Student.name FROM Club LEFT OUTER JOIN Student ON Student.id = Club.leader;

SELECT address, avg(id) FROM Student GROUP BY address;

SELECT cast('2018-12-15 09:08:32' AS DATE);
SELECT cast('2018-12-15 09:08:32' AS DATETIME);
SELECT cast(1.657 AS SIGNED INTEGER), convert(1.567, SIGNED INTEGER);
SELECT now(), str_to_date('02/03/18', '%m/%d/%Y');
SELECT now(), str_to_date('2012-02-03', '%Y-%m-%d');
SELECT concat('aaa', 'bbb', 'ccc');
SELECT concat_ws('--', 'aaa', 'bbb', 'ccc');
SELECT address, min(name) FROM Student GROUP BY address;
SELECT address, group_concat(name) as 'student name' FROM Student GROUP BY address;
SELECT address, if(address = '강릉' or address = '서울', '*', ''), group_concat(name) as 'student name' 
FROM Student GROUP BY address;
SELECT name, ifnull(leader, 'absent') FROM Club;

SHOW VARIABLES LIKE '%commit%';
SELECT @@autocommit;

START TRANSACTION;
UPDATE Student SET name = '111' WHERE id = 1;
SELECT * FROM Student WHERE id = 1;
COMMIT;
ROLLBACK;

CREATE TABLE `mycnlife`.`Grade` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
  `midterm` TINYINT UNSIGNED NOT NULL DEFAULT 0,
  `finalterm` TINYINT UNSIGNED NOT NULL DEFAULT 0,
  `enroll` INT UNSIGNED NULL,
  PRIMARY KEY (`id`));
ALTER TABLE `mycnlife`.`Grade` ADD CONSTRAINT FOREIGN KEY (enroll) REFERENCES Enroll (id) 
ON DELETE RESTRICT ON UPDATE RESTRICT;
DESCRIBE Grade;
SHOW CREATE TABLE Grade;

INSERT INTO Grade(midterm, finalterm, enroll) 
SELECT ceil((0.5 + rand() / 2) * 100), mod(id, 50) + 50, id FROM Enroll; 
TRUNCATE TABLE Grade;
SELECT * FROM Grade ORDER BY enroll;
SELECT count(*), (SELECT count(*) FROM Enroll) FROM Grade;
SELECT min(midterm), max(midterm) FROM Grade;
SELECT min(finalterm), max(finalterm) FROM Grade;

SELECT Grade.finalterm, (Grade.midterm + Grade.finalterm) AS total, 
	(Grade.midterm + Grade.finalterm) / 2 AS average FROM Grade;
SELECT Student.name AS 'student', Subject.name AS 'subject', Grade.finalterm, (Grade.midterm + Grade.finalterm) AS total, 
	(Grade.midterm + Grade.finalterm) / 2 AS average FROM Grade 
    INNER JOIN Enroll ON Grade.enroll = Enroll.id
    INNER JOIN Student ON Enroll.student = Student.id
    INNER JOIN Subject ON Enroll.subject = Subject.id;
SELECT sub.* FROM (
	SELECT Student.name AS 'student', Subject.name AS 'subject', Grade.finalterm, (Grade.midterm + Grade.finalterm) AS total, 
		(Grade.midterm + Grade.finalterm) / 2 AS average FROM Grade 
		INNER JOIN Enroll ON Grade.enroll = Enroll.id
		INNER JOIN Student ON Enroll.student = Student.id
		INNER JOIN Subject ON Enroll.subject = Subject.id
) AS sub ORDER BY sub.average DESC;
SELECT sub.*, (CASE 
	WHEN average >= 90 THEN 'A'
	WHEN average >= 80 THEN 'B'
	WHEN average >= 70 THEN 'C'
	WHEN average >= 60 THEN 'D'
	ELSE 'F' END) AS rating
    FROM (
		SELECT Student.name AS 'student', Subject.name AS 'subject', Grade.finalterm, (Grade.midterm + Grade.finalterm) AS total, 
			(Grade.midterm + Grade.finalterm) / 2 AS average FROM Grade 
			INNER JOIN Enroll ON Grade.enroll = Enroll.id
			INNER JOIN Student ON Enroll.student = Student.id
			INNER JOIN Subject ON Enroll.subject = Subject.id
) AS sub ORDER BY sub.average DESC;

DESCRIBE Grade;
DESCRIBE Enroll;
SELECT subject, count(*) FROM Enroll GROUP BY subject;
SELECT max(Subject.name), avg(Grade.midterm + Grade.finalterm) / 2 As 'average', count(*) FROM Grade 
	INNER JOIN Enroll ON Grade.enroll = Enroll.id
	INNER JOIN Subject ON Enroll.subject = Subject.id
	GROUP BY Subject.id;
SELECT Student.name, (Grade.midterm + Grade.finalterm) AS total FROM Grade
	INNER JOIN Enroll ON Grade.enroll = Enroll.id
	INNER JOIN Student ON Enroll.student = Student.id
    WHERE Subject.name = '국어' ORDER BY total DESC, Grade.finalterm DESC LIMIT 2;

SELECT max(Subject.name) 'subject', avg(Grade.midterm + Grade.finalterm) / 2 As 'average', count(*) AS 'student no.', 
	(SELECT Student.name FROM Grade
	INNER JOIN Enroll ON Grade.enroll = Enroll.id
	INNER JOIN Student ON Enroll.student = Student.id
    WHERE Subject.id = Enroll.subject 
    ORDER BY (Grade.midterm + Grade.finalterm) DESC, Grade.finalterm DESC LIMIT 1) AS 'best'
    FROM Grade 
	INNER JOIN Enroll ON Grade.enroll = Enroll.id
	INNER JOIN Subject ON Enroll.subject = Subject.id
	GROUP BY Subject.id;

CREATE TABLE `mycnlife`.`ClubMember` (
  `id` SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `club` SMALLINT UNSIGNED NOT NULL,
  `student` INT UNSIGNED NOT NULL,
  `level` TINYINT(1) UNSIGNED NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`));
  
ALTER TABLE ClubMember ADD CONSTRAINT FOREIGN KEY (club) REFERENCES Club(id);
ALTER TABLE ClubMember ADD CONSTRAINT FOREIGN KEY (student) REFERENCES Student(id);

DESCRIBE Club;
SELECT * FROM Club;
SELECT id, leader, 2 FROM Club WHERE leader IS NOT NULL;
ALTER TABLE Club DROP COLUMN leader;
SHOW CREATE TABLE Club;

ALTER TABLE Club DROP FOREIGN KEY club_ibfk_1;
ALTER TABLE Club DROP INDEX leader;
ALTER TABLE Club DROP COLUMN leader;

SELECT * FROM Club;
SELECT * FROM ClubMember;
SELECT id, ceil(rand() * 1000), 2 FROM Club LIMIT 4;
INSERT INTO ClubMember(club, student, level) SELECT id, ceil(rand() * 1000), 2 FROM Club LIMIT 4;
SELECT Club.id, Student.id FROM Club, Student;
SELECT Club.id, Student.id FROM Club, Student ORDER BY rand() LIMIT 150;
TRUNCATE TABLE ClubMember;
INSERT INTO ClubMember(club, student) SELECT Club.id, Student.id FROM Club, Student ORDER BY rand() LIMIT 150;
UPDATE ClubMember SET level = 1 ORDER BY rand() LIMIT 4;
SELECT * FROM ClubMember WHERE level = 1;
UPDATE ClubMember SET club = 4 WHERE id = 101;

SELECT count(*) FROM ClubMember;
SELECT club, count(*) FROM ClubMember GROUP BY club;
SELECT club, student FROM ClubMember GROUP BY club, student HAVING count(*) > 1;
SELECT count(DISTINCT Student) FROM ClubMember;

CREATE TABLE `mycnlife`.`Department` (
  `id` SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(45) NOT NULL,
  `profess` SMALLINT UNSIGNED NULL,
  `student` INT UNSIGNED NULL,
  PRIMARY KEY (`id`));

ALTER TABLE Department ADD CONSTRAINT FOREIGN KEY (profess) REFERENCES Profess(id) ON DELETE SET NULL;
ALTER TABLE Department ADD CONSTRAINT FOREIGN KEY (student) REFERENCES Student(id) ON DELETE SET NULL;

INSERT INTO Department(name, profess) SELECT '국문학과', id FROM Profess ORDER BY rand() LIMIT 1;
INSERT INTO Department(name, profess) SELECT '영문학과', id FROM Profess ORDER BY rand() LIMIT 1;
INSERT INTO Department(name, profess) SELECT '물리학과', id FROM Profess ORDER BY rand() LIMIT 1;
INSERT INTO Department(name, profess) SELECT '역사학과', id FROM Profess ORDER BY rand() LIMIT 1;
INSERT INTO Department(name, profess) SELECT '사회학과', id FROM Profess ORDER BY rand() LIMIT 1;
SELECT * FROM Department;

ALTER TABLE Student ADD COLUMN department SMALLINT UNSIGNED;
ALTER TABLE Student ADD CONSTRAINT FOREIGN KEY (department) REFERENCES Department(id);

SELECT * FROM Student;
UPDATE Student SET department = (SELECT id FROM Department ORDER BY rand() LIMIT 1);
UPDATE Department SET student = (SELECT id FROM Student WHERE department = Department.id ORDER BY rand() LIMIT 1);

SELECT Department.id, Department.name, Department.student, Student.department AS 'StuDept' FROM Department 
	INNER JOIN Student ON Student.id = Department.student;
SELECT * FROM Student WHERE id = 55;

CREATE TABLE `mycnlife`.`Classroom` (
  `id` SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`id`));

INSERT INTO Classroom(name) SELECT concat(ceil(rand(id) * 10), id, '호') FROM Subject;
SELECT * FROM Classroom;
ALTER TABLE Subject ADD COLUMN classroom SMALLINT UNSIGNED;
ALTER TABLE Subject ADD CONSTRAINT FOREIGN KEY (classroom) REFERENCES Classroom(id);
SELECT * FROM Subject;
UPDATE Subject SET classroom = (11 - id);