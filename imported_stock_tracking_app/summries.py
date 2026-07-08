import requests
import json
import sys

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e4b" 

def summarize_content_with_ollama(content_to_summarize: str, prompt_instruction: str) -> str:
    """
    Sends content and instructions to the Ollama API to generate a summary.

    Args:
        content_to_summarize: The long text that needs summarizing.
        prompt_instruction: The specific instructions given to the model 
                             (e.g., "Summarize this content into three bullet points").

    Returns:
        The generated summary text, or an error message.
    """
    
    # 1. Craft the full prompt payload
    # It's best practice to give the model clear instructions AND the data.
    full_prompt = f"{prompt_instruction}\n\n--- CONTENT TO SUMMARIZE ---\n{content_to_summarize}"
    
    # 2. Define the request payload for the Ollama API
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,  # Set to False to wait for the full response
        "options": {
            "temperature": 0.2, # Lower temperature = more factual and reliable
            "num_predict": 512 
        }
    }

    print(f"⚙️ Sending request to Ollama ({MODEL_NAME})...")

    try:
        # 3. Make the API request
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status() # Will throw an HTTPError for bad status codes
        
        # 4. Parse the JSON response
        response_json = response.json()
        
        # The generated text is contained within the 'response' key
        summary = response_json.get("response", "").strip()
        return summary

    except requests.exceptions.ConnectionError:
        return "❌ ERROR: Could not connect to Ollama. Please ensure the Ollama server is running in the background."
    except requests.exceptions.HTTPError as e:
        return f"❌ ERROR: HTTP request failed. Status Code: {e.response.status_code}. Check your model name or API URL."
    except requests.exceptions.Timeout:
        return "❌ ERROR: Request timed out. The model might be processing a very large request."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"


# =================================================================
#                   EXAMPLE USAGE
# =================================================================

# 1. The complex content you want summarized (e.g., the SEACOR Marine text)
article_content = """
On May 20, 2026, SEACOR Marine Holdings Inc. (the “Company”), as parent guarantor, 
and SEACOR Marine Foreign Holdings Inc., as borrower and wholly-owned subsidiary of the 
Company (“SMFH”), entered into a letter agreement (“Letter Agreement”) for the purposes 
of modifying that certain credit agreement, dated as of November 27, 2024, among the 
Company, SMFH, certain other wholly-owned subsidiaries of the Company, as subsidiary 
guarantors, an affiliate of EnTrust Global, as lender, Kroll Agency Services Limited, 
as facility agent, and Kroll Trustee Services Limited, as security trustee (the “2024 Credit Agreement”). 
... (The rest of the original text)
""" 

# 2. The specific instructions for the model (Prompt Engineering)
#summary_prompt = "Please provide a concise, professional summary of this document for an executive audience. Focus on the financial changes and the purpose of the agreement."

# 3. Call the function
#summary = summarize_content_with_ollama(article_content, summary_prompt)

# 4. Print the result
#print("\n" + "="*80)
#print("✅ SUCCESSFULLY GENERATED SUMMARY:")
#print(summary)
#print("="*80)
