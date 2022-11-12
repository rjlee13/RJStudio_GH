



"""
Intro to Docker
 (▰˘◡˘▰)


This video covers basic Docker commands 🐣

I am going to assume that you have successfully installed Docker onto your machine! 🚀


I decided to pull postgres image from Docker Hub
https://hub.docker.com/_/postgres 
It was downloaded over 1 BILLION times (1B+)!!


Please 🌟PAUSE🌟 the video any time you want to study the commands used.
"""













### pull postgres image (latest version)
docker pull postgres


### let's check postgres image
docker images


### run postgres!  
docker run --name rjlee-postgres \
-e POSTGRES_PASSWORD=rj-postgres \
postgres 


	"notice logs pop up! you can use another terminal window 
							OR 
	you can let logs run in the background by using -d flag!"

-d: Run container in background and print container ID


### first quit the container that is running
control+C 


### run postgres with -d flag
docker run --name rjl-postgres -d \
-e POSTGRES_PASSWORD=rj-postgres \
postgres 

	
	"now logs do NOT pop up!"


### check postgres container we just started
docker ps -a


	"we have one container called 'rjl-postgres' running 
	 with -d flag
	 
	 we have another container called 'rjlee-postgres' which 
	 I exited to stop seeing logs earlier"












### PSQL, the front-end of PostgreSQL
docker exec -it rjl-postgres psql -U postgres


### list of all database
\l

### Create DB called rjdb
CREATE DATABASE rjdb;
\l


	"now we have rjdb database!"


### simple SQL command
SELECT NOW();


### to quit 
\q


### another way to use PSQL: first start bash
docker exec -it rjl-postgres bin/bash


### and then, PSQL
psql -U postgres















### we can ALSO
### assign username 'rj-postgres' from docker run 
### create database 'rj-data'     from docker run 
docker run --name rj-postgres -d \
-e POSTGRES_USER=rj-postgres \
-e POSTGRES_PASSWORD=rj-postgres \
-e POSTGRES_DB=rj-data \
postgres 



### Interactive console
### notice username 'rj-postgres'
### notice database 'rj-data'
docker exec -it rj-postgres psql -U rj-postgres rj-data


### list database
\l











### check all postgres containers 
docker ps -a


### remove all containers (to free up machine's resource!)
docker rm rj-postgres --force
docker rm rjl-postgres --force
docker rm rjlee-postgres --force


### check if all containers are gone
docker ps -a

	"No longer see any containers!"













"""
This is the end of "Intro to Docker" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""










