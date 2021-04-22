SHOW DATABASES;
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
  `id` SMALLINT(0) UNSIGNED NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(30) NOT NULL,
  `likecnt` INT NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`));

CREATE TABLE `mycnlife`.`Subject` (
  `id` SMALLINT(0) UNSIGNED NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(45) NOT NULL,
  `profess` SMALLINT(0) NULL,
  PRIMARY KEY (`id`));

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
  `subject` SMALLINT(0) UNSIGNED NOT NULL,
  `student` INT UNSIGNED NOT NULL,
  PRIMARY KEY (`id`));

ALTER TABLE Enroll ADD CONSTRAINT FOREIGN KEY (subject) REFERENCES Subject (id) ON DELETE CASCADE;
ALTER TABLE Enroll ADD CONSTRAINT FOREIGN KEY (student) REFERENCES Student (id) ON DELETE CASCADE;

DESC Enroll;
SELECT * FROM Club;

INSERT INTO Club (name, leader) VALUES ('요트부', 100);
INSERT INTO Club (name, leader) VALUES ('음악부', 200);
INSERT INTO Club (name, leader) VALUES ('미술부', 300);

SELECT *, Student.name as 'student name' FROM Club INNER JOIN Student on Club.leader = Student.id;
SELECT Club.*, Student.name as 'student name' FROM Club INNER JOIN Student on Club.leader = Student.id;



