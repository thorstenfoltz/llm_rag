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
* Execute `make all` to build the docker images and start docker compose. Pay attention, depending on your settings you must adjust the makefile and add simply `sudo` before each command. If you don't have `make` installed and don't want to install it, move on with the commands within the `Makefile`. You can open it with any text editor.
* Within the file `rag.py` you find at the top configuration options. Due to limit time during the development, everything is stored in a single file. Currently, data is fetched and ingested into the database. This will take some time (approx 30-60 Minutes) depening on your system. If you want to re-run it later, make sure to change in the file `load = yes` to `load = no`. Then the download process will be skipped. You'll not ingest the data twice, if you don't do that, but it still needs the same amount of time.
* Within the same file you'll see `pages = 1`, which simply means the first page and all reports from there are fetched. You can add more pages, but be aware this will need a lot of time and during the development the web site of the WHO doesn't react from time to time.
* If you want to re-run, simply execute `make run_rag`

## Architecture
![](https://github.com/username/repository/blob/master/arch.png)

## How does it work?
It works in the following way:
1. Anything in docker to make sure, that all requirements are fulfilled
2. A Postgres database, version 17, with pgvector addon is set up.
3. The script uses the browser Chrome to go the web site of the WHO. This is needed, because when new files are loaded, the URL doesn't change. So, it is required to go to next side by using Javascript.
4. The script fetches all links in a Python set, to make sure, every links is only loaded once.
5. Looping over the URL of each file to download it.
6. Each file is splitted into different chunks. To increase the usage, the length is variable, not fixed. Using regular expressions to find headers and bullet points. If a paragraph is too long it is divided into separate paragraphs with some overlap. Python library nltk is used for it.
7. All chunks are vectorized and loaded into the Postgres database. The table contains:
    1. id: each chunk gets an automatically created id
    2. section_content: the content (text) of the chunk
    3. type: is it a header or a paragraph
    4. vector: the vector of the chunk created by OpenAI's `text-embedding-3-small`
    5. hash: a unique hash to prevent that equal chunks are loaded twice
    6. group_id: an id which shows which chunks are coming from the same file
    7. timestamp: just the point at which the data was inserted
8. In this step, the user can ask any question. The question is vectorized by the same model as the chunks.
9. The vector of the user's query is compared in four different approaches:
    1. Cosine Similarity
    2. Euclidean Distance by using FAISS (Facebook AI Similarity Search)
    3. Dot product also by FAISS
    4. Canberra Distance
10. The top 5 results of each method from 9 is taken and receives points. 5 for the best answer, 4 for the second best and so on. Afterwards the top three answers are taken and shown (only the id, group id and the score).
11. The best answer is taken as context and send together with the user's question agains OpenAI's model `gpt-4o`
12. An answer is received, the user can decide whether he leaves or make a new request.

## Limitations
Please be aware, that the model accept any question, since `gpt-4o` provides the answer. That means, you'll receive an answer, but it doesn't mean it's taken any information from the database. The data in the database refers to health and disease topics (it's the WHO).

## Improvements
There are many improvments which could be implemented:
1. Set up an UI
2. Set up some Monitoring
3. Devide the script into several scripts
4. Loading data from other sources
5. Using linters
6. Store user's questions and the corresponding answers in a database, for reuse
7. Clean the script

And many more. Perhaps I will implement and adjust more in the future, but for this project time was very limited.
