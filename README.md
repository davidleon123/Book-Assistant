# Submission NUMHACK-2024 build category
# Book Assistant

## Application Description

It is a web-based application that provides an easy-to-use RAG applications to allow students to learn class material more quickly.

The application can be accessed at [Book Assistant](https://ai-demo.fr)

This submission demonstrates that it is possible to enhance class material at a very low-cost to the great benefit of students around the world.

From the point of view of the student, they have a very simple yet clear user interface to retrieve the content they need to study. They can ask a question in the web application and they will get :

- references to the relevant passages of the material they have to study

- an answer to their question based on the retrieved passages.

The emphasis is here on the cost effectiveness and ease-of-use of the system.

For demonstration purposes, the application is currently using two freely available javascript books. Questions should thus be about Javascript. If a question is not about Javascript or if the answer cannot be found in the source material, the RAG will answer that it does not know.
Questions end with a word of encouragement for the student to keep learning.

## Technical solution

The solution is based on open-source components. It is using chroma as a vector database, langchain to build the RAG chain and Django as a front end.

The solution is hosted in a virtual private server on the cloud.

As it stands, documents are uploaded by placing them in the `book upload` directory, and making a call to the `load` function of the `src/db_handler.py`file.

In future, this PoC will be extended to allow the teacher to upload the material through the web application frontend.

## Cost analysis

This is meant as a low-cost solution for schools or institutions scaling to hundreds of students.

As the application is built on free Open Source components, there is no upfront cost to deploying the system.

The cost for the virtual private server is around 3€/month, which is about 3 cents per student per month on the assumption of the systeme being used in an institution of 100 students.



## future work

- In the web application, add a sign up logging in for students and teachers
- Add in the web application, the possibility for teachers to upload new materials
- Add the selection of different databases for different classes 
- Try different LLMs in the RAG chain
- Benchmarking load and response time of the server with increased number of users

