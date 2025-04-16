# Nexus AI Customer Support Chatbot - Capstone Project

This repository contains the code for the Nexus AI Customer Support Chatbot, developed as a Capstone project for the Google Gen AI Intensive Course (2025Q1).

## Overview

Nexus AI is an intelligent assistant designed for "Nexus Creative Studio" (a fictional web agency). It uses Retrieval-Augmented Generation (RAG) with Google Gemini and Function Calling to answer user questions based on a provided PDF knowledge base and perform actions like providing booking links.

## Features & Gen AI Capabilities

*   **RAG:** Answers grounded in the `Nexus Creative Studio.pdf` document.
*   **Function Calling:** Can provide a booking link when asked.
*   **Embeddings & Vector Search:** Uses Sentence Transformers and FAISS.
*   **Query Expansion:** Leverages Gemini to improve search recall.
*   **Structured Output:** Generates predictable JSON responses (internally).
*   **Gen AI Evaluation:** Includes code for LLM-based response assessment.
*   **Interactive UI:** Deployed using Gradio on Hugging Face Spaces.

## Live Demo

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://ndumisombili-nexus-customer-chatbot.hf.space/)

Access the live, interactive demo here: https://huggingface.co/spaces/NdumisoMbili/Nexus-Customer-Chatbot

## Blog Post

Read more about the project development, challenges, and learnings here: https://www.linkedin.com/pulse/from-pdf-chatbot-my-google-gen-ai-intensive-capstone-project-mbili-7oyzf

## Local Setup (Optional)

1.  Clone the repository: `git clone <your-repo-url>`
2.  Navigate to the project directory: `cd <your-repo-name>`
3.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
4.  Install dependencies: `pip install -r requirements.txt`
5.  Set the Gemini API Key as an environment variable:
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY_HERE" # Linux/macOS
    # set GEMINI_API_KEY=YOUR_API_KEY_HERE    # Windows CMD
    # $env:GEMINI_API_KEY="YOUR_API_KEY_HERE" # Windows PowerShell
    ```
    **(IMPORTANT: Do NOT commit your actual API key to Git!)**
6.  Ensure `Nexus Creative Studio.pdf` is in the main directory.
7.  Run the Gradio app: `python app.py`

## Capstone Project

This project fulfills the requirements for the Google Gen AI Intensive Course Capstone (2025Q1).
