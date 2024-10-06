# LLM Project Datazoomcamp

## Problem Description
Target of this project in short: Download pdf files, read them in chunks into a database, build a vector on each chunk and query the database by using LLM.

Many times reports are published in a certain format: pdf. While this is a good format for humans it is difficult to analyze automatically or simply to get certain information from it, if you have many files, but don't know where to search.
In this project I use pdf files from the [WHO](https://www.who.int/europe/publications/i). The task is to download reports published by the WHO, transform each report into meaningful parts, store them in a Postgres database, vectorized them and give the possiblity to ask questions by using a LLM which get context from the database.

## What is needed?
* Docker and Docker Compose
* An API key from OpenAI
* Make should be installed, but it is possible to run anything without it

## How to install?
* Clone this repository
* Make sure you fulfill the requirements from above
* Create a file `.env` and add the following: `OPENAI_API_KEY=your_key`
* Store `.env` at the root of the repository (the same place where you see all the other files)
* Execute `make all` to build the docker images and start docker compose. Pay attention, depending on your settings you must adjust the makefile and add simply `sudo` before each command.
* Within the file `rag.py` you find at the top configuration options. Due to limit time during the development, everything is stored in a single file. Currently, data is fetched and ingested into the database. This will take some time (approx 30-60 Minutes) depening on your system. If you want to re-run it later, make sure to change in the file `load = yes` to `load = no`. Then the download process will be skipped. You'll not ingest the data twice, if you don't do that, but it still needs the same amount of time.
* Within the same file you'll see `pages = 1`, which simply means the first page and all reports from there are fetched. You can add more pages, but be aware this will need a lot of time and during the development the web site of the WHO doesn't react from time to time.
* If you want to re-run, simply execute `make run_rag`

## Architecture
![](https://github.com/thorstenfoltz/llm_rag/blob/master/arch.png)