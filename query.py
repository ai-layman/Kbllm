import os
import requests
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
import tkinter as tk
from tkinter import ttk

class ChatBoxUI(tk.Tk):
    def __init__(self, docsearch, chain, embeddings):
        super().__init__()

        self.title("Chatbot")
        self.geometry("400x500")

        self.docsearch = docsearch
        self.chain = chain
        self.embeddings = embeddings
        self.query_history = []
        self.response_history = []

        self.create_widgets()

    def create_widgets(self):
        self.chat_frame = ttk.Frame(self)
        self.chat_frame.pack(expand=True, fill=tk.BOTH)

        self.chat_text = tk.Text(self.chat_frame, wrap=tk.WORD, state="disabled")
        self.chat_text.pack(expand=True, fill=tk.BOTH)

        self.entry_frame = ttk.Frame(self)
        self.entry_frame.pack(fill=tk.X)

        self.user_entry = ttk.Entry(self.entry_frame)
        self.user_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.user_entry.bind("<Return>", self.process_query)

        self.send_button = ttk.Button(self.entry_frame, text="Send", command=self.process_query)
        self.send_button.pack(side=tk.RIGHT)

    def process_query(self, event=None):
        query = self.user_entry.get()
        if not query:
            return

        self.query_history.append(query)
        self.query_history = self.query_history[-5:]

        self.update_chat(f"User: {query}\n")

        # Fetch the most similar documents
        top_k = 5
        docs = self.docsearch.similarity_search(query, top_k=top_k)

        result = self.chain.run(input_documents=docs, question=query)

        self.response_history.append(result)
        self.response_history = self.response_history[-5:]

        if self.moderate_content(result):
            self.update_chat(f"Bot: {result}\n")
        else:
            self.update_chat("Bot: [Content moderated]\n")

        self.user_entry.delete(0, tk.END)

    def update_chat(self, message):
        self.chat_text.configure(state="normal")
        self.chat_text.insert(tk.END, message)
        self.chat_text.configure(state="disabled")
        self.chat_text.see(tk.END)

    def moderate_content(self, content):
        api_key = os.environ.get('OPENAI_API_KEY')
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
        data = {
            'input': content,
        }

        response = requests.post('https://api.openai.com/v1/moderations', headers=headers, json=data)
        moderation_result = response.json()

        return not moderation_result["results"][0]["flagged"]

def main():
    # Load environment variables
    load_dotenv()

    # Initialize Pinecone
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pinecone_api_env = os.environ.get('PINECONE_API_ENV')
    index_name = os.environ.get('INDEX_NAME')
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_api_env)

    # Initialize embeddings, docsearch, LLM, and chain
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Initialize Pinecone with Pinecone Index and embeddings
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")

    # Create and run the ChatBox UI
    chatbox_ui = ChatBoxUI(docsearch, chain, embeddings)
    chatbox_ui.mainloop()

    # Deinitialize Pinecone
    pinecone.deinit()

if __name__ == "__main__":
    main()


