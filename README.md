<p align="center">
  <img src="readme img/backgroundPencil.png">
</p>

# Large Language Models for Analysis of Qualitative Data in Research (Backend)
This project, developed as a master's project under the guidance of Dr. Mateusz Dolata, holds a web application where users can chat with their own interview data. The backend is written using the Django framework and relies on Pinecone for document vectorization and GPT-3.5 and GPT-4 for answer generation.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Technology](#technology)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
Welcome to Qwizz! 🚀

Qwizz, short for "Q" for qualitative interview data and "wizz" as a reference to its chatbot functionality, is the brainchild of Richard Specker, Ivelin Ivanov, and Kilian Sennrich, three master students from the University of Zurich (UZH). With guidance from Dr. Mateusz Dolata of the Department of Informatics - Information Management Research Group, we've developed this user-friendly web application to address the challenges faced by researchers when handling large volumes of qualitative interview data.

Over a series of interviews (reqirement engineering), we've learnt about the struggle of sifting through a mountain of interviews (sometimes more than 50!), trying to extract meaningful insights. Qwizz steps in to assist with this daunting task! It doesn't automate the entire analysis process, but rather acts as your trusty companion, guiding you through your interviews, helping you find your information quickly, and inspiring new ideas.

Qwizz simplifies the process of interacting with your interview data through a chatbot interface. It extracts and presents relevant content to you, allows you to explore relevant interview passages, and ultimately empowers you to make the most of your research journey.

Give Qwizz a try, and let it be your partner in the exploration and understanding of your interview data. Click [here](https://qwizz-frontend.ivelin.info/) to get started. Happy researching! 📚💡

## Installation
Follow these installation steps:

**Clone the Repository**: Start by cloning the Qwizz repository to your local machine. You can do this by running the following command in your terminal:
```shell
git clone https://github.com/LLM-for-Qualitative-Data-Analysis/dockerized-backend
```

**Install Docker**: If you haven't already, make sure you have Docker installed on your system. You can download and install Docker from Docker's [official website](https://www.docker.com/).

**Navigate to your Qwizz Directory**: Change your working directory to the location where you've cloned the Qwizz repository using the cd command:
```shell
cd your/local/qwizz/directory
```

**Build the Docker Container**: Run the following command to build the Docker container for Qwizz:
```shell
docker-compose build
```

**Start the Qwizz backend**: Once the container is built, you can start Qwizz with the following command:
```shell
docker-compose up
```

That's it! Qwizz should now be up and running, and you can access it through your web browser on http://localhost:8000 . If you encounter any issues during installation, make sure you've followed these steps correctly and have Docker configured properly on your system.

## Usage

 In the following you'll discover a selection of the essential features of Qwizz.

![Projects](readme%20img/projects.png)
**Create a New Project!** Projects help you organize your work. They provide you with a dedicated space to cluster related chats and organize your documents. Add a new project by clicking on the New Project button. </br></br>

![Chat](readme%20img/chat.png)
**Create a New Chat and Start Your Conversation!** Within the Chats tab, you can categorize your project into separate chats to keep discussions well-structured. Every chat possesses its own memory, guaranteeing accurate context and personalized interactions. Chats are time-stamped upon creation, and can be renamed using the three dots adjacent to the title. </br></br>

![Summary](readme%20img/summary.png)
**Gain a Comprehensive Overview!** Click the Background Information option in the dropdown menu, which can be accessed by clicking the three dots in the document upload modal. This action will open a modal providing you with a more comprehensive overview of the document. You'll find a word cloud and an AI-generated summary of the entire document. </br></br>

![Notes](readme%20img/notes.png)
**Write Down Your Thoughts!** Click the Notes icon in the icon bar on the left and open the notes section. Clicking on a specific note opens up the editing modal on the right. Don't forget to save your changes. </br></br>

![Settings](readme%20img/settings.png)
**Personalize Your Responses!** Access the Settings to tailor your responses to your preferences. You have the flexibility to modify the language model, adjust the temperature, and define the answer length to meet your specific needs.

This is only an assortment of the functions of the software, a more detailed list of the features can be found on the website's index page.

## Technology
- **Deployment**: The backend is deployed on DigitalOcean, which ensures our  high computational demands.
- **Web Server**: Gunicorn with its WSGI HTTP server was used to serve the Django backend, providing efficient and reliable web service.
- **Backend Framework**: The backend is built using the Django Python framework, offering a robust and feature-rich environment for building web applications.
- **Orchestration Tool**: Haystack was employed as an LLM orchestration tool to enhance preprocessing and build a state-of-the-art LLM application.
- **File Storage**: We used AWS File Storage, including Amazon CloudFront and Amazon S3, for efficient and secure file storage and distribution.
- **API Integration**: The backend utilizes the following APIs:
    - *Pinecone*: For advanced search and similarity matching.
    - *OpenAI*: For incorporating AI-driven features and capabilities.
    - *QuickChart*: For generating dynamic and interactive charts and graphs.

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear messages.
4. Push your branch to your fork on GitHub.
5. Create a pull request to the main repository's `main` branch.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.txt) file for details.

**[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)**

## Contact
- **Developer**: Richard Jörg Specker: richardjoerg.specker@uzh.ch
- **Developer**: Ivelin Ivanov: ivelin.ivanov@uzh.ch
- **Developer**: Kilian Sennrich: kilian.sennrich@uzh.ch
- **Supervisor**: Dr. Mateusz Dolata: mateusz.dolata@uzh.ch

Don't hesitate to reach out to us if you have any questions!
