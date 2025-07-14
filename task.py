import cohere
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

co = cohere.Client("6jeEUAzQKv3WkJuz5OBSUh7YGTc3ZpbhOHOwfV4J")  
embedder = SentenceTransformer("all-MiniLM-L6-v2")
faqs = [
    {"q": "What is Resolute.ai?", "a": "Resolute.ai is a data science and AI solutions company."},
    {"q": "How do I reset my password?", "a": "Click 'Forgot Password' on the login page and follow the email instructions."},
    {"q": "Where is Resolute.ai located?", "a": "Resolute.ai operates remotely with team members across the globe."},
    {"q": "How can I contact support?", "a": "You can contact support by emailing support@resolute.ai."},
    {"q": "What technologies does Resolute.ai use?", "a": "We use Python, machine learning, AI models, and cloud computing."},
    {"q": "How do I apply for an internship at Resolute.ai?", "a": "Visit our careers page or email careers@resolute.ai with your resume."},
    {"q": "What projects do interns work on?", "a": "Interns contribute to real-world AI, NLP, and data science projects."},
    {"q": "Does Resolute.ai offer remote internships?", "a": "Yes, we offer remote internships to selected candidates."},
    {"q": "What is a RAG system?", "a": "RAG stands for Retrieval-Augmented Generation, combining search with language models."},
    {"q": "Which programming languages are preferred?", "a": "We mostly use Python, with occasional use of SQL, JS, and Bash."},
    {"q": "Is prior ML experience required to join?", "a": "It's preferred but not mandatory; we value curiosity and learning."},
    {"q": "How often are team meetings held?", "a": "We hold weekly standups and monthly knowledge-sharing sessions."},
    {"q": "Can I contribute to open-source projects?", "a": "Yes, employees and interns are encouraged to contribute to OSS."},
    {"q": "What are embeddings?", "a": "Embeddings are vector representations of text used in search and NLP."},
    {"q": "How do I get API access?", "a": "You can request API keys by filling out the internal developer form."},
    {"q": "Is there a document for onboarding?", "a": "Yes, new members receive a PDF guide and a Notion onboarding page."},
    {"q": "What is Cohere used for?", "a": "We use Cohere for both text generation and semantic embeddings."},
    {"q": "How do I join the internal Slack?", "a": "You'll receive an invite to Slack after HR onboarding is complete."},
    {"q": "What are the core values of Resolute.ai?", "a": "Curiosity, innovation, integrity, and collaboration."},
    {"q": "How do I raise a bug or feature request?", "a": "Use the internal issue tracker or report via Slack to the dev team."}
]
answers = [faq["a"] for faq in faqs]
answer_embeddings = embedder.encode(answers)
index = faiss.IndexFlatL2(len(answer_embeddings[0]))
index.add(np.array(answer_embeddings).astype("float32"))

def for_llm(question):
    response = co.generate(
        model="command-xlarge",
        prompt=question,
        max_tokens=300
    )
    return response.generations[0].text.strip()

def for_rag(question):
    question_emb = embedder.encode([question])
    _, I = index.search(np.array(question_emb).astype("float32"), k=1)
    context = answers[I[0][0]]
    prompt = f"Use this context to answer:\n\n{context}\n\nQuestion: {question}"
    response = co.generate(
        model="command-xlarge",
        prompt=prompt,
        max_tokens=300
    )
    return response.generations[0].text.strip()
