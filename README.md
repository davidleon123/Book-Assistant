# Submission NUMHACK-2024 build category
# Project Name: Book Assistant

## Application Description

We built a web-based application that provides an easy-to-use RAG to allow students to learn class material more quickly.

One of the benefits of using the RAG instead of having students use an LLM chatbot like chatGPT is that the answers are based solely on the course material and the risk of having the LLM hallucinate is thus minimal.

It can be accessed at [http://ai-demo.fr](http://ai-demo.fr)

Please note that https access has not yet been activated and you should access the application with http only.

This submission demonstrates that it is possible to enhance class material at a very low-cost to the great benefit of students around the world.

From the point of view of the student, they have a very simple yet clear user interface to retrieve the content they need to study. They can ask a question in the web application and they will get :

- references to the relevant passages of the material they have to study

- an answer to their question based on the retrieved passages.

The emphasis is here on the cost effectiveness and ease-of-use of the system.

For demonstration purposes, the application is currently using two freely available javascript books. Questions should thus be about Javascript. If a question is not about Javascript or if the answer cannot be found in the source material, the RAG will answer that it does not know.

Answers end with a word of encouragement for the student to keep learning.

## Technical Category

Build

## Challenge Topic

Education: Help low resources students access and study from teachers' books. 

## Technical solution

The solution is based on open-source components. It is using chroma as a vector database, langchain to build the RAG chain and Django as a front end. The RAG chain is currently made of the chroma database chained to a call to the GPT3.5 API. In future, the API could possibly be replace with a small edge LLM.

The solution is hosted in a virtual private server on the cloud.

As it stands, documents are uploaded by placing them in the `book upload` directory, and making a call to the `load` function of the `src/db_handler.py`file.

In future, this PoC will be extended to allow the teacher to upload the material through the web application frontend.

## Cost analysis

This is meant as a low-cost solution for schools or institutions scaling to hundreds of students.

As the application is built on free Open Source components, there is no upfront cost to deploying the system.

The cost for the virtual private server is around 5$ per month, which is about 5 cents per student per month on the assumption of the systeme being used in an institution of 100 students.

For the LLM part of the RAG chain, assuming the use of the GPT 3.5 API at prices around 0.00015$ per 1K tokens, a use of around 500 tokens per request and 50 requests per student per day, the cost per student per month would thus be around 10 cents.

A ballpark estimate of the system cost (server + API usage) would thus be around 15 cents per student per month. The total cost for a school with 100 students would thus be around 15$ per month.


## future work

- In the web application, add a sign up and logging in for students and teachers

- Add in the web application, the possibility for teachers to upload new materials

- Add the selection of different databases for different classes 

- Try if a small LLM model could run in the server instead of using an LLM API

- Benchmarking load and response time of the server with increasing numbers of users

## Project Repository URL

https://github.com/davidleon123/Book-Assistant.git

## Deployment URL

[http://ai-demo.fr](http://ai-demo.fr)

Please note that https access has not yet been activated and you should access the application with http only.

## Project Video File

[video](https://drive.google.com/file/d/1aOG-6w_q0o40pn7BRT7aXSj9X1mwIMrh/view?usp=sharing)

## Team members

[Dea María Léon](https://github.com/DeaMariaLeon)

[David Leon](https://github.com/davidleon123)

## Tech stack

Chroma

Langchain

GPT3.5 API

Django

Linux

nginx

gunicorn

Simplecss

css

html

Python