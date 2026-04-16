import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-pro')

def ask_gemini():
    prompt = input("You: ")
    response = model.generate_content(prompt)
    print(f"Gemini: {response.text}")

if __name__ == "__main__":
    ask_gemini()