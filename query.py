import os
import requests
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import pinecone
import tkinter as tk
from tkinter import ttk

# ChatBox User Interface
class ChatBoxUI(tk.Tk):
    # Initialize ChatBox; Pass required objects for documents search, LLM, and embeddings
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

    # Create ChatBox UI elements and set-up ChatBox configuration
    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.chat_frame = ttk.Frame(self)
        self.chat_frame.grid(row=0, column=0, sticky="nsew")
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1)

        self.chat_text = tk.Text(self.chat_frame, wrap=tk.WORD, state="disabled", font=("Helvetica", 14))
        self.chat_text.grid(row=0, column=0, sticky="nsew")

        self.entry_frame = ttk.Frame(self)
        self.entry_frame.grid(row=1, column=0, sticky="nsew")
        self.entry_frame.grid_columnconfigure(0, weight=1)

        self.user_entry = ttk.Entry(self.entry_frame)
        self.user_entry.grid(row=0, column=0, sticky="nsew")
        self.user_entry.bind("<Return>", self.process_query)

        self.send_button = ttk.Button(self.entry_frame, text="Send", command=self.process_query)
        self.send_button.grid(row=0, column=1, sticky="e")

    # Process user query, search for similar documents, and run QA chain
    def process_query(self, event=None):
        query = self.user_entry.get()
        if not query:
            return

        self.query_history.append(query)
        self.query_history = self.query_history[-5:]

        self.update_chat(f"User: {query}\n")
        self.show_thinking_message()

        # Fetch the most similar documents
        top_k = 5
        docs = self.docsearch.similarity_search(query, top_k=top_k)

        result = self.chain.run(input_documents=docs, question=query)

        self.response_history.append(result)
        self.response_history = self.response_history[-5:]

        if self.moderate_content(result):
            self.update_chat(f"Bot: {result}\n", replace=True)
        else:
            self.update_chat("Bot: [Content moderated]\n", replace=True)

        self.user_entry.delete(0, tk.END)

    # Update ChatBox with messages
    def apply_tags(self, start, end, tag_name, **options):
        self.chat_text.tag_configure(tag_name, **options)
        self.chat_text.tag_add(tag_name, start, end)

    def show_thinking_message(self):
        self.chat_text.configure(state="normal")
        start_index = self.chat_text.index(tk.END + "-1c linestart")
        self.chat_text.insert(tk.END, '|')  # Insert the blinking cursor
        self.chat_text.mark_set(tk.INSERT, start_index)  # Set the position of the blinking cursor
        self.chat_text.configure(state="disabled")
        self.chat_text.see(tk.END)
        self.chat_text.update_idletasks()

    def update_chat(self, message, replace=False):
        self.chat_text.configure(state="normal")

        if replace:
            start_index = self.chat_text.index(tk.END + "-1c linestart")
            self.chat_text.delete(start_index, tk.END)  # Remove the blinking cursor
            self.chat_text.insert(tk.END, "\n")  # Add a newline character

        start_index = self.chat_text.index(tk.END + "-1c linestart")
        self.chat_text.insert(tk.END, message)
        end_index = self.chat_text.index(tk.END + "-1c linestart")

        if message.startswith("User:"):
            self.apply_tags(start_index, end_index, "user_message")
        elif message.startswith("Bot:"):
            self.apply_tags(start_index, end_index, "bot_message", background="#f0f0f0")

        self.chat_text.configure(state="disabled")
        self.chat_text.see(tk.END)
        self.chat_text.update_idletasks()

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
    chain = load_qa_with_sources_chain(llm, chain_type="stuff")

    # Create and run the ChatBox UI
    chatbox_ui = ChatBoxUI(docsearch, chain, embeddings)
    chatbox_ui.mainloop()

    # Deinitialize Pinecone
    pinecone.deinit()

if __name__ == "__main__":
    main()


