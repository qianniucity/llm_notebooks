# import libraries
import streamlit as st

from classifier.language_classifier import LanguageDetector
from encoder.encoder import Encoder
from generator.generator import Generator
from retriever.vector_db import VectorDatabase
from translator.translator import Translator

# init classes
generator = Generator()
encoder = Encoder()
vectordb = VectorDatabase(encoder.encoder)
translator = Translator()
lang_classifier = LanguageDetector()

# create the app
st.title("Welcome")

# create the message history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# render older messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# render the chat input
prompt = st.chat_input("Enter your message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # render the user's new message
    with st.chat_message("user"):
        st.markdown(prompt)

    # render the assistant's response
    with st.chat_message("assistant"):
        # get the product id and the customer question
        ID = prompt.split("|")[0]
        QUERY = prompt.split("|")[1]

        # detect customer language to reply back in same language
        user_detected_language = lang_classifier.detect_language(QUERY)

        # retrieve context about the product
        context = vectordb.retrieve_most_similar_document(QUERY, k=4, id=ID)

        # convert all context into english for our LLM
        english_context = []
        for doc in context:
            detected_language = lang_classifier.detect_language(doc)
            if detected_language != "en_XX":
                doc = translator.translate(doc, detected_language, "en_XX")
            english_context.append(doc)
        context = "\n".join(english_context)

        # translate customer query to english
        if user_detected_language != "en_XX":
            QUERY = translator.translate(QUERY, user_detected_language, "en_XX")

        # generate anwser with our LLM based on customer query and context
        answer = generator.get_answer(context, QUERY)
        # if the customer language is not english, then translate it
        if user_detected_language != "en_XX":
            answer = translator.translate(answer, "en_XX", user_detected_language)
        # show answer to customer
        st.markdown(answer)

    # add the full response to the message history
    st.session_state.messages.append({"role": "assistant", "content": answer})
