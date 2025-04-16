# --- Core Libraries ---
import os
import time
import json
import re
import html  # For escaping HTML in references
import numpy as np
import faiss
import gradio as gr
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from collections import Counter

print("Libraries imported.")

# --- Constants ---
PDF_FILENAME = "Nexus Creative Studio.pdf"  # Expect this file in the same directory
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 512
OVERLAP = 100
TOP_K_RESULTS = 3

# --- API Key Configuration (Read from Environment Variable for Hugging Face Spaces) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GENERATION_MODEL_NAME = None # Will be set later

if not GEMINI_API_KEY:
    print("FATAL ERROR: GEMINI_API_KEY environment variable not found.")
    print("Please set the GEMINI_API_KEY secret in your Hugging Face Space settings.")
    # In a real deployment, you might want the app to fail gracefully here
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API Key configured.")

        # --- Select Gemini Model (Robust Selection) ---
        print("Listing available Gemini models...")
        available_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            print(f"Found models: {available_models}")

            # Prioritize preferred models
            preferred_models = [
                'models/gemini-1.5-flash-latest',
                'models/gemini-1.0-pro',
                'models/gemini-pro',
            ]

            for model_name in preferred_models:
                if model_name in available_models:
                    GENERATION_MODEL_NAME = model_name
                    break

            # Fallback if preferred models aren't available
            if not GENERATION_MODEL_NAME and available_models:
                GENERATION_MODEL_NAME = available_models[0]

        except Exception as e:
            print(f"Warning: Error listing models ({e}). Falling back.")

        # Final fallback if listing failed or no suitable model found
        if not GENERATION_MODEL_NAME:
             GENERATION_MODEL_NAME = 'models/gemini-1.5-flash-latest' # Default guess

        print(f"Using Generation Model: {GENERATION_MODEL_NAME}")

    except Exception as e:
        print(f"Error during Gemini configuration or model selection: {e}")
        GENERATION_MODEL_NAME = None # Mark as unavailable if setup fails


# --- Function Definitions ---

def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file."""
    text = ""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n" # Add newline between pages
        print(f"Extracted text from {pdf_path} (approx. {len(text)} chars).")
        return text
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return ""

def chunk_text_with_overlap(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Chunks text into smaller pieces with overlap."""
    if not text: return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # Try to end chunk at sentence boundary if reasonable
        if end < text_len and (end - start) > overlap:
            potential_end = text.rfind('. ', start + overlap, end)
            if potential_end != -1:
                 end = potential_end + 1 # Include the period
        chunks.append(text[start:end].strip())
        # Move start for the next chunk
        start += chunk_size - overlap
        # Prevent infinite loops
        if start >= end and end < text_len:
             start = end
    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk]
    print(f"Chunked text into {len(chunks)} chunks.")
    return chunks

def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    """Loads the Sentence Transformer embedding model."""
    try:
        print(f"Loading embedding model: {model_name}...")
        # Consider adding device='cuda' if deploying on GPU hardware, otherwise default is fine
        model = SentenceTransformer(model_name)
        print(f"Embedding model '{model_name}' loaded (Dimension: {model.get_sentence_embedding_dimension()}).")
        return model
    except Exception as e:
        print(f"FATAL ERROR loading embedding model '{model_name}': {e}")
        return None

def create_faiss_index(chunks, embedding_model):
    """Creates a FAISS index for the given text chunks."""
    if not chunks or not embedding_model:
        print("Skipping FAISS index creation: No chunks or model.")
        return None
    try:
        print(f"Creating FAISS index for {len(chunks)} chunks...")
        embeddings = embedding_model.encode(chunks, show_progress_bar=False).astype('float32')
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        # Using IndexFlatIP for cosine similarity with normalized vectors
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        print(f"FAISS index created (Dimension: {dimension}, Size: {index.ntotal}).")
        return index
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None

def expand_query(query, model_name=GENERATION_MODEL_NAME):
    """Expands the user query using an LLM."""
    if not model_name or not GEMINI_API_KEY:
        print("Query expansion skipped: Generation model not available.")
        return [query] # Return original query as a list

    prompt = f"""Rewrite the following user query to be more detailed and potentially include synonyms or related concepts for better information retrieval. Return only the rewritten query, without any preamble.
Original query: "{query}"
Rewritten query:"""
    try:
        expansion_model = genai.GenerativeModel(model_name)
        response = expansion_model.generate_content(prompt)
        expanded_query = response.text.strip()
        # Basic validation
        if expanded_query and expanded_query.lower() != query.lower():
            # print(f"Expanded query: {expanded_query}") # Optional logging
            return [query, expanded_query] # Return list with original and expanded
        return [query] # Return only original if expansion failed or was same
    except Exception as e:
        print(f"Warning: Query expansion failed ({e}). Using original query.")
        return [query]

def calculate_keyword_score(query_terms, chunk_text):
    """Calculates a simple keyword overlap score."""
    chunk_terms = Counter(re.findall(r'\w+', chunk_text.lower()))
    score = sum(min(count, chunk_terms.get(term, 0)) for term, count in query_terms.items())
    return score

def hybrid_search(queries, index, chunks, embedding_model, top_k=TOP_K_RESULTS, keyword_weight=0.3):
    """Performs hybrid search using semantic (FAISS) and keyword scoring."""
    if not index or not chunks or not embedding_model:
        print("Hybrid search skipped: Index, chunks, or model not available.")
        return []

    combined_results = {}
    for query in queries:
        # 1. Semantic Search (FAISS)
        try:
            query_embedding = embedding_model.encode([query]).astype('float32')
            faiss.normalize_L2(query_embedding)
            # Search more results initially to allow for reranking
            distances, faiss_indices = index.search(query_embedding, k=max(top_k, index.ntotal // 10)) # Search ~10% or top_k
        except Exception as e:
            print(f"Warning: FAISS search failed for query '{query}': {e}")
            distances, faiss_indices = [[]], [[]] # Empty results on error

        # 2. Keyword Search Prep
        query_terms = Counter(re.findall(r'\w+', query.lower()))
        max_possible_kw_score = sum(query_terms.values()) or 1 # Avoid division by zero

        # --- Combine Scores ---
        # Iterate through all chunks to calculate keyword scores
        for i, chunk_text in enumerate(chunks):
            # Get semantic score if this chunk was found by FAISS for this query
            semantic_score = 0.0
            if i in faiss_indices[0]:
                 idx_in_results = np.where(faiss_indices[0] == i)[0][0]
                 semantic_score = float(distances[0][idx_in_results]) # Similarity score

            # Calculate keyword score
            keyword_score = calculate_keyword_score(query_terms, chunk_text)
            normalized_kw_score = keyword_score / max_possible_kw_score

            # Calculate combined score (adjust weighting as needed)
            combined_score = (1.0 - keyword_weight) * semantic_score + keyword_weight * normalized_kw_score

            # Store or update results, keeping the best score for each unique chunk index
            if i not in combined_results or combined_score > combined_results[i]['combined_score']:
                combined_results[i] = {
                    'chunk': chunk_text,
                    'combined_score': combined_score
                }

    # --- Rank and Return Top K ---
    if not combined_results:
        print("No results found after combining scores.")
        return []

    sorted_results = sorted(combined_results.values(), key=lambda x: x['combined_score'], reverse=True)

    # print(f"Hybrid search reranked {len(sorted_results)} potential results.") # Optional logging
    return sorted_results[:top_k]

def format_support_response(query, results, model_name=GENERATION_MODEL_NAME):
    """Generates a structured support response using an LLM based on retrieved context."""
    if not model_name or not GEMINI_API_KEY:
        return {"error": "Response generation skipped: Generation model not available or API key missing."}
    if not results:
        # Provide a standard response if no context was found
        return {
            "answer": "I couldn't find specific information relevant to your query in my knowledge base. Could you please try rephrasing or asking a different question?",
            "steps": [],
            "urgency": 2, # Default urgency
            "references": []
        }

    # --- Prepare Context ---
    context = "\n\n---\n\n".join([result['chunk'] for result in results])

    # --- Basic Query Type Detection (for potential persona adjustment) ---
    query_lower = query.lower()
    persona = "Nexus's support agent" # Default persona
    if re.search(r'price|cost|plan|payment|R\d+', query_lower):
        persona = "Nexus's pricing specialist"
    elif re.search(r'how (to|do)|guide|tutorial|steps', query_lower):
         persona = "Nexus's technical guide"
    elif re.search(r'login|password|reset|account', query_lower):
        persona = "Nexus's account support agent"

    # --- Create the Prompt ---
    prompt = f"""
    You are {persona}. Your task is to answer the customer's query based *only* on the provided context. Do not add information not present in the context.

    CONTEXT FROM KNOWLEDGE BASE:
    ---
    {context}
    ---

    CUSTOMER QUERY: "{query}"

    INSTRUCTIONS:
    1. Carefully analyze the CONTEXT and the CUSTOMER QUERY.
    2. Generate a concise and helpful answer to the QUERY using *only* the information found in the CONTEXT. If the answer involves multiple points or steps derived from the text, use newline characters (`\n`) within the answer string for better readability where appropriate.
    3. If the context doesn't contain the answer, clearly state that you couldn't find the specific information in the provided context. Do not make assumptions or provide external knowledge.
    4. Extract any actionable steps the user might need to take, based *only* on the context. If no steps are mentioned, provide an empty list ([]).
    5. Determine an urgency level (1=Low, 3=Medium, 5=High) based on the query type (e.g., general info low, pricing medium, account access high). Default to 2 if unsure.
    6. List short, relevant quotes or phrases (max 15 words each) from the CONTEXT that directly support your answer as references. If no direct quotes support the answer, provide an empty list ([]).
    7. Format your entire response *strictly* as a single JSON object with the following keys: "answer", "steps", "urgency", "references". Ensure the output is valid JSON.

    JSON RESPONSE:
    """

    # --- Generate Response using Gemini with JSON mode ---
    try:
        generation_model = genai.GenerativeModel(model_name)
        response = generation_model.generate_content(
            prompt, # Pass prompt directly as first argument
            generation_config={"response_mime_type": "application/json"}
        )

        # --- Parse the JSON response ---
        # Clean potential markdown code block fences
        response_text = response.text.strip().strip("```json").strip("```").strip()

        result = json.loads(response_text)

        # Basic validation of expected keys
        if not all(k in result for k in ["answer", "steps", "urgency", "references"]):
             raise ValueError("Generated JSON missing required keys (answer, steps, urgency, references).")

        # print("Successfully generated structured response.") # Optional logging
        return result

    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: Failed to parse LLM response: {e}")
        print(f"Raw response text (first 500 chars): {response.text[:500]}...")
        return {"error": "Failed to parse the generated response as JSON.", "raw_response": response.text}
    except Exception as e:
        print(f"Error during response generation: {e}")
        # Attempt to get raw response text even if other errors occurred
        raw_text = "Failed to retrieve raw response text."
        try: raw_text = response.text
        except: pass
        return {"error": f"An unexpected error occurred during response generation: {str(e)}", "raw_response": raw_text}


# --- Global Variables Initialization (Load models and data once on startup) ---
# These run when the script starts on Hugging Face Spaces
embedding_model = load_embedding_model()
extracted_text = extract_text_from_pdf(PDF_FILENAME) if os.path.exists(PDF_FILENAME) else None
all_chunks = chunk_text_with_overlap(extracted_text) if extracted_text else []
faiss_index = create_faiss_index(all_chunks, embedding_model) if all_chunks and embedding_model else None

# Check if initialization was successful
INITIALIZATION_SUCCESSFUL = all([embedding_model, extracted_text, all_chunks, faiss_index, GENERATION_MODEL_NAME])
if not INITIALIZATION_SUCCESSFUL:
    print("\nWARNING: One or more components failed to initialize. The application might not function correctly.")
    print(f"Status: EmbeddingModel={bool(embedding_model)}, PDFText={bool(extracted_text)}, Chunks={bool(all_chunks)}, FAISSIndex={bool(faiss_index)}, GenModel={bool(GENERATION_MODEL_NAME)}")


# --- Main Processing Function ---
def process_customer_query(query):
    """Processes a customer query end-to-end through the RAG pipeline."""
    print(f"\nProcessing query: '{query}'") # Log query processing start

    if not INITIALIZATION_SUCCESSFUL:
        # Return error if setup failed
        return {"error": "System initialization failed. Please check server logs."}

    # --- 1. Expand Query ---
    expanded_queries = expand_query(query)

    # --- 2. Retrieve Context (Hybrid Search) ---
    # Assuming single index 'Nexus' for now
    search_results = hybrid_search(expanded_queries, faiss_index, all_chunks, embedding_model, top_k=TOP_K_RESULTS)

    # --- 3. Generate Response ---
    generated_response = format_support_response(query, search_results)

    # Evaluation step is removed for the deployed app's main flow

    print("Query processing complete.") # Log query processing end
    return generated_response


# --- Gradio Interface Definition (with UI/UX fixes AND SyntaxError fix) ---
def gradio_interface_formatter(query):
    """
    Wrapper function for the Gradio interface.
    Calls the main pipeline and formats the output cleanly for display,
    ensuring compatibility with light and dark themes and fixing f-string issue.
    """
    # Call the backend processing function
    generated_response = process_customer_query(query)

    # --- Format the output as HTML ---
    # Use minimal inline styling to allow Gradio themes to control appearance.
    output_html = f"<div style='padding: 10px; line-height: 1.6;'>"

    # Display the original query for context
    output_html += f"<p style='color: grey; font-size: 0.9em;'>Query: \"{html.escape(query)}\"</p><hr style='margin: 10px 0;'>"

    # Handle potential errors during processing
    if "error" in generated_response:
        # Simple error display
        output_html += f"""
        <div style='color: #D8000C; background-color: #FFD2D2; border: 1px solid #D8000C; padding: 10px; border-radius: 5px;'>
            <strong>Error:</strong> {html.escape(generated_response['error'])}
            <p style='font-size: 0.8em; color: #555;'><i>Details: {html.escape(generated_response.get('raw_response','N/A')[:200])}...</i></p>
        </div>
        """
    else:
        # --- CORRECTED ANSWER FORMATTING ---
        # 1. Get the raw answer
        raw_answer = generated_response.get('answer', 'No answer generated.')
        # 2. Escape for HTML safety FIRST
        escaped_answer = html.escape(raw_answer)
        # 3. Replace actual newline characters (\n) with HTML line breaks (<br>) AFTER escaping
        formatted_answer = escaped_answer.replace('\n', '<br>')
        # 4. Insert the fully processed string into the f-string
        output_html += f"<p>{formatted_answer}</p>"
        # --- END OF CORRECTION ---

        # Display Suggested Steps if they exist
        steps = generated_response.get('steps', [])
        if steps:
            output_html += "<p style='margin-top: 15px;'><strong>Suggested Steps:</strong></p>"
            output_html += "<ol style='margin-top: 5px; padding-left: 25px;'>" # Indent list
            for step in steps:
                output_html += f"<li>{html.escape(step)}</li>"
            output_html += "</ol>"

        # Display References if they exist
        references = generated_response.get('references', [])
        if references:
            output_html += "<p style='margin-top: 15px;'><strong>References from knowledge base:</strong></p>"
            output_html += "<ul style='margin-top: 5px; padding-left: 25px; font-size: 0.9em; color: grey;'>"
            for ref in references:
                safe_ref = html.escape(ref)
                output_html += f"<li><i>\"{safe_ref}\"</i></li>"
            output_html += "</ul>"

    output_html += "</div>" # Close the main div
    return output_html

# --- Create Gradio Interface ---
# Only create the functional interface if initialization was successful
if INITIALIZATION_SUCCESSFUL:
    print("Essential components loaded. Creating Gradio interface.")
    interface_title = "Nexus Customer Support AI"
    interface_description = "Get answers to your questions about Nexus Creative Studio's services, pricing, and processes. Powered by Google Gemini and RAG."
    interface_examples = [
        ["What kind of websites do you build?"],
        ["What are the pricing plans?"],
        ["How long does it take to build a website?"],
        ["Do I need technical knowledge?"],
        ["How much does the Premium plan cost?"]
    ]

    demo = gr.Interface(
        fn=gradio_interface_formatter, # Use the updated formatter
        inputs=gr.Textbox(lines=3, placeholder="Ask your question about Nexus Creative Studio here..."),
        outputs=gr.HTML(),
        title=interface_title,
        description=interface_description,
        examples=interface_examples,
        allow_flagging='never', # If using older Gradio versions
        # flagging_mode='never' # If using newer Gradio versions
        theme=gr.themes.Default() # Use default theme for better dark/light mode handling
    )
else:
    # Create a dummy interface that displays an error if setup failed
    print("WARNING: Gradio interface cannot be created due to initialization errors. Creating an error display interface.")
    def error_interface(*args):
        log_check_msg = "Please check the application logs on the Hugging Face Space for details."
        return f"<div style='padding:20px;'><h2 style='color:red'>Application Initialization Failed</h2><p>{log_check_msg}</p><p>Essential components like the PDF file ({PDF_FILENAME}), embedding model, or API configuration might have failed to load.</p></div>"

    demo = gr.Interface(
        fn=error_interface,
        inputs=gr.Textbox(label="Input (disabled)"), # Input doesn't matter here
        outputs=gr.HTML(),
        title="Nexus AI - Initialization Error",
        description="The application could not start correctly."
        )

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio App...")
    # server_name="0.0.0.0" allows access from network, useful for Docker but not needed for HF Spaces direct launch
    # share=True is for local tunneling, not needed for HF Spaces
    demo.launch()