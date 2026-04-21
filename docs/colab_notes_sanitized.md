# Project 03 - Chat with your documents using advanced RAG

> In this project, we will learn how to create a more advanced RAG pipeline that can:
* ask questions about a document that has been read, as if it were a chat with the file itself.
* consult more than one reference at the same time.
* understand the context of past messages, using the conversation history as a reference to formulate the response

And an interface will also be built for this application.
Therefore, we can reuse part of the code from the previous project and add new features.

## [ ! ] How to run locally
To run the code for this project locally, follow the instructions to install the required dependencies using the commands below. You can use the same installation commands.

## Installation and Configuration

Here we will first load all the functions we used in the previous project and a few others. Among them, FAISS (a vectorstore in the same style as Chroma, which we used in previous classes on RAG) and also other functions necessary for implementing a RAG pipeline that understands the context of conversations.

Remember: we can reuse part of the code we created in project 02.
So if you want, you can make a copy and make the modifications from there.


!pip install -q streamlit langchain
!pip install -q langchain_community langchain-huggingface langchain_ollama langchain_openai

### FAISS Installation

Before importing, we need to install FAISS. It is not installed by default in Colab. Therefore, we can use the same command here and locally to install it:

`pip install -q faiss-cpu`

You can also install `faiss-gpu` if you want to use the GPU-optimized version. For the sake of simplicity, we will use the default CPU version.



!pip install -q faiss-cpu

Then use `import faiss` in your application and also import `FAISS` inside langchain library

import faiss
from langchain_community.vectorstores import FAISS

### Installing PyPDFLoader

We will use PyPDFLoader to read PDF files in our application. This will be explained in detail in the appropriate section.
To use it, we first need to install the library with the command below.

!pip install pypdf

### Other installations

Just like in the previous project, we will install dotenv again (in a local environment there is no need to run the installation again, but here in Colab, since it is a new session, we need it) and also localtunnel (remember that this is not necessary in a local environment).

!pip install -q python-dotenv
!npm install -q localtunnel

%%writefile .env
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key_here
OPENAI_API_KEY=your_openai_api_key_here



## (code explanations - step by step)

First, we will do all the necessary imports

### Creating the sidebar in the interface

Let's create an element in our interface where we will upload files. In this case, we will create a sidebar, where there will be a field to upload PDF files to be processed in the application.

If no file is uploaded, a message is displayed asking the user to upload a file and interrupts execution until a file is uploaded.

We can add this code snippet right after the declared functions

Explanations:

* `uploads = st.sidebar.file_uploader(...)` - Creates a file upload component in the interface's sidebar, allowing the user to upload multiple PDF files. Here we specify the parameter to accept multiple documents and limit it to PDFs only (at the moment we are interested in programming to accept only this extension, but later we can increase support, allowing us to accept an infinite number of different extensions)

* `if not uploads:` - Checks if no file was uploaded.

* `st.info` - displays the message in the interface
* `st.stop()` - Stops the execution of the code after the message, preventing the rest of the application from running without files loaded.

## Creates side panel in the interface
uploads = st.sidebar.file_uploader(
    label="Upload files", type=["pdf"],
    accept_multiple_files=True
)
if not uploads:
    st.info("Please send some file to continue!")
    st.stop()

### Pipeline for Indexing and Retrieval

This part basically consists of the indexing and retrieval steps that we learned previously.

Just to remind you:

1. Load the content of the PDF or other file/media/website/etc.
2. Division into Chunks: divide all the contents of the documents into small pieces, or chunks.
3. Storage and Transformation into Embeddings: These chunks are transformed into embeddings, which are vector representations of the texts. The embeddings are stored in a vector database.
4. Use of Retriever: The vector database provides a retriever that searches for the most relevant chunks based on a similarity algorithm.
5. Generation: Joining the context to the prompt and generating the final result (the inference will not be placed in this function, so we will do the code later)

Therefore, since the indexing and retrieval pipeline is essentially the same as our previous RAG application, we can reuse the code.

In the previous project we did everything in the same function (`model_response()`), and in this project we separated it into two (the retriever function that we are seeing now; and the function that will return the chain, as will be shown below). We do this to have greater flexibility as the pipeline is more complex now

For the indexing and retrieval pipeline we gathered all of this inside a `config_retriever()` function that will accept the documents sent as a parameter

*(explanations below)*

def config_retriever(uploads):
    # Load
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Store
    vectorstore = FAISS.from_documents(splits, embeddings)

    vectorstore.save_local('vectorstore/db_faiss')

    # Retrieve
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k':3, 'fetch_k':4}
    )

    return retriever

> The code here has some differences and additional details compared to the previous one. We will list them below

#### - Loading documents (Reading files)

This code block below prepares documents to be indexed and used in RAG. The code loads PDFs provided by the user, stores them in a temporary directory, and processes them so that they can be used to create embeddings and retrieve information with retrieval.

Previously we used WebBaseLoader to read a web page. Now, since we want to read a PDF, we will use PyPDFLoader. To use it, you need to import `from langchain_community.document_loaders import PyPDFLoader` (it is already in the code block with all the imports, at the beginning of Colab).

> Check out other DocumentLoaders here: https://python.langchain.com/docs/integrations/document_loaders/

"""docs = []
temp_dir = tempfile.TemporaryDirectory()
for file in uploads:
    temp_filepath = os.path.join(temp_dir.name, file.name)
    with open(temp_filepath, "wb") as f:
        f.write(file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    docs.extend(loader.load())"""

#### - Choosing an appropriate embedding model

* One thing that can significantly improve results is choosing a better or more appropriate embedding model for the language.

* In the experiments we conducted, these showed a crucial difference compared to the model we were using in the first examples with RAG, which did not return such good results because this model was not as prepared for our language.

* And why might multilingual models be interesting? Not only because you may want to focus on tests with content in other languages, but because the content read may contain texts in multiple languages (that you maybe even doesn't expect), and if the embedding is not done correctly for these languages then the LLM will not return good results. Thus, it is better to develop a RAG pipeline that consumes several sources from different languages.

* Therefore, multilingual embedding models can be essential for RAG systems, allowing robust retrieval and generation of information in different languages.
* Selecting the right model for your RAG system is a crucial decision that impacts not only the quality of the response, but also resource utilization and scalability. By carefully considering whether it works well for specific languages or tasks, you can find the best one for your needs.    

> **Which models to choose**

* Open Source Models
 * The BGE models in HuggingFace are currently considered the best open source embedding models. The BGE model is created by BAAI - Beijing Academy of Artificial Intelligence. BAAI is a private non-profit organization engaged in AI research and development.

 * In recent benchmarks, the top performing open source model was [BGE-M3](https://huggingface.co/BAAI/bge-m3). The model has the same context length as the OpenAI models (8K), and is approximately 2.2 GB in size.

 * There are other lighter (or even larger) alternatives such as [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) and its reduced version, [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).

 * **`[!]`** We still recommend BGE-M3 at the moment. But if it is too heavy for your setup, you can use one of the two mentioned above, such as BAAI/bge-small-en-v1.5, which may not be as good as those designed for multiple languages, but they tend to work quite well and are a much lighter solution that maintains much of the quality (at least, this was the case in several tests we performed).

* Proprietary models
 * can be an excellent idea if you don't want to bother with this.
 * There are from OpenAI, Google, Cohere, Anthropic, etc.
 * For example, for OpenAI, just change to the `OpenAIEmbeddings()` method.
 * Using proprietary models can be slightly more practical, in exchange for paying very few cents every million tokens (check the pricing page on the company's website)

> **How to change the model**

* To use it, just replace the HuggingFaceEmbeddings function parameter in our code, from `sentence-transformers/all-mpnet-base-v2` to `BAAI/bge-m3`, or whatever you want.

* Note: Here we are reusing the same function that we already know for loading embeddings, but there are alternatives that can even be more efficient depending on the configuration, such as [FastEmbedEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.fastembed.FastEmbedEmbeddings.html), which is based on FastEmbed, from [Qdrant](https://github.com/qdrant), a company who is a reference in AI and vector stores.
 * We keep it the same as Hugging Face to avoid installing yet another library and because it also tends to be fast, but if you notice that embedding could be faster (which will be more noticeable if you are loading very large text files) then you can test this function, just install it and change the method, doing its import first (check [documentation](https://python.langchain.com/docs/integrations/text_embedding/fastembed/)).

> **Where to find more models**

* search for `multilingual` (or by searching the name of your desired language) in hugging face, within the sentence-similarity models category

* https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=trending&search=multilingual

* https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=trending&search=portuguese

* Or in leaderboards, just like we've already seen leaderboards for llms, there are also embedding models. One example is the MTEB Leaderboard on Hugging Face, which provides an up-to-date list of proprietary and open-source embedding models, complete with performance statistics on various tasks such as retrieval and summarization - https://huggingface.co/spaces/mteb/leaderboard

`Note:` As with LLMs, it can be important to stay informed about new embedding models to reevaluate and update your choices accordingly.


> **More details on open source vs. proprietary**

* OpenAI’s recent pricing revision has made access to its API significantly more affordable.

* However, cost-effectiveness is not the only factor to consider. Other aspects such as latency, privacy, and control over data processing flows may also be important.

* Open-source models offer the advantage of full control over the data, improving privacy and personalization. On the other hand, OpenAI’s API may have latency issues, resulting in long response times.

* In short, it is not always a simple choice.

* Proprietary solutions can generally be more efficient and interesting for those who prioritize convenience, especially if privacy is not a major concern.

* Open-source embedding models are an attractive option due to the advantages discussed above, combining performance with greater control over the data.

All the text discussed above regarding this topic is related to the code snippet below

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

#### - Choosing a vector database

To show an alternative, we will use FAISS instead of the Chroma DB we used before. We will see how easy it is to switch (when there is direct integration with langchain)

About FAISS
* Facebook AI Similarity Search (FAISS) is a library for efficient similarity search and dense vector clustering.
* It contains algorithms that search in sets of vectors of any size, even those that probably do not fit in RAM.
* It is a widely chosen option because it is scalable and works well for much larger datasets.
* The real advantage of FAISS comes when dealing with large datasets that require fast access to relevant information. It retrieves similar vectors efficiently, which can be very relevant even for other types of systems such as facial recognition that depend on fast data comparison, empowering sectors such as security and surveillance with cutting-edge features.

> https://github.com/facebookresearch/faiss

Other solutions
* Other open source solutions besides [Chroma](https://www.trychroma.com/) and FAISS are for example [Weaviate](https://python.langchain.com/docs/integrations/vectorstores/weaviate/) and [Milvus](https://python.langchain.com/docs/integrations/vectorstores/milvus/).
* And there are also other solutions like [VectorstoreIndexCreator](https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.html) from LangChain, which is less scalable but may be more practical for some. See the LangChain vector stores page for full details

Proprietary solutions:
* One option that is recommended in this case is [**Pinecone**](https://www.pinecone.io/), an optimized cloud-based vector database.
* Highly scalable due to cloud infrastructure.
* May be preferred for its simplicity and ease of use.

* Pinecone is designed to efficiently store and retrieve dense vector embeddings, making it ideal for enhancing LLMs with long-term memory and improving their performance in tasks such as natural language processing.
* It offers fast data retrieval, ideal for chatbots, and includes a free tier for storing up to 100,000 vectors (check their page because pricing may change in the future).

So while these open source vector databases exist, using such a solution can be practical and simple, and can run efficiently on any machine.

We will use FAISS because it is open source, but if you want to use Pinecone just go to the website https://www.pinecone.io/ generate a token, add it to the environment variables and load it using Langchain's Pinecone method instead of using the FAISS method (see [documentation here](https://python.langchain.com/docs/integrations/vectorstores/pinecone/) to copy the command)

#!pip install -q faiss-cpu
"""vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local('vectorstore/db_faiss')"""

#### - Change in the retriever

Just to show how you could use a different retrievr method, we will use a different search algorithm: MMR - Maximum marginal relevance retrieval. It has already been explained in the RAG colab, but as a reminder:
 * MMR selects by relevance and diversity among the retrieved documents to avoid passing through duplicate context, ensuring a balance between relevance and diversity in the retrieved items.

Since we are using MMR, in addition to the `k` parameter (used to define the number of documents returned by the retriever), we also have the option of defining another parameter, `fetch_k`.

* This `fetch_k` parameter defines how many documents are retrieved before applying the MMR algorithm (default value: 20).
* A higher value of fetch_k (e.g.: 20-50) can generate more diverse and relevant results, since MMR will have more documents to choose from.
* However, higher values of fetch_k increase the computational cost, since MMR needs to be applied to a larger set of documents.
* The optimal value of fetch_k depends on the size and quality of your vector store, and the desired balance between result quality and performance.

* Compared to the `k` parameter (number of documents returned):

* A smaller value of k (e.g. 1-5) is ideal when you need a few highly relevant documents, such as in simple Q&A tasks.

* A larger value of k (e.g. 10-20) is useful when you want to provide more context or options to the user, such as in a search engine or recommendation system.

Therefore, the best values depend on the specific use case and requirements of your application.

In general, it is recommended to start with smaller values of k and fetch_k, gradually increasing them as performance and relevance of results improve. Experimenting with different values and monitoring the impact on application behavior can help you find the optimal balance.

"""retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k':3, 'fetch_k':4}
)"""

The other codes within the function we created for the retriever remain the same as in the previous RAG example, so they were not commented on above.

With this, we have completed the indexing and retrieval pipeline. We can return to the function with `return retriever`, since we want to use this function later in our code and thus assign the retriever to a variable.

---
### Advanced chain for conversation

Now we will be making modifications to the chain that we have already implemented previously and are familiar with. We will do it a little differently this time.

As we saw in the previous project, the user's query may require context from the conversation to be understood.

In many question and answer (Q&A) applications - as is the case with this project now with RAG for conversation with documents - we want to allow the user to have a fluid conversation and to be able to refer to what was recently said, which means that the RAG pipeline needs some kind of "memory" of previous messages and some logic to incorporate them into its current thinking.

In project 1, we implemented a way to read the history, which worked very well, but now for RAG we need a more advanced pipeline if we want better results.

In short, what we need to change in the logic:

* Prompt - update our prompt to support message history.

* Contextualizing questions - add a subchain that takes the user's last question and reformulates it in the context of the chat history. You can simply think of this as building a new history-aware retriever.

In other words, while before we had:
* query -> retriever

Now we will have:
* (query, chat history) -> LLM -> reformulated query -> retriever

This way, LLM will be able to understand when a question is asked in the sequence and needs to know what the previous message was in order to understand what it is referring to (these questions asked in the sequence are also called "Follow up questions")
* For example, if we type "talk about company XYZ" and then ask "when was it founded?", we want the model to understand that "it" refers to "company XYZ". And to do this, it reformulates the question based on the history, hence this additional step.
* If you really know that you do not need this behavior, then you can use the same RAG pipeline seen previously, creating the chain that way.
 * But since we want a more advanced example, let's learn this mode now

All the code for this advanced chain is shown below, together in the `config_rag_chain()` function

🔗 [This diagram here](https://python.langchain.com/v0.2/assets/images/conversational_retrieval_chain-5c7a96abe29e582bc575a0a0d63f86b0.png) basically shows what we are doing in this function.

*- Explanations below -*

def config_rag_chain(model_class, retriever):

    ### Loading the LLM
    if model_class == "hf_your_huggingface_key_here":
        llm = model_hf_your_huggingface_key_here()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()

    # Prompt definition
    if model_class.startswith("hf"):
        token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        token_s, token_e = "", ""

    # Contextualization prompt
    context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."

    context_q_system_prompt = token_s + context_q_system_prompt
    context_q_user_prompt = "Question: {input}" + token_e
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", context_q_user_prompt),
        ]
    )

    # Chain for contextualization
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=context_q_prompt
    )

    # Q&A Prompt
    qa_prompt_template = """You are a helpful virtual assistant answering general questions.
Use the following bits of retrieved context to answer the question.
If you don't know the answer, just say you don't know. Keep your answer concise.
Answer in English. \n\n
    Question: {input} \n
    Context: {context}"""

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

    # Configure LLM and Chain for Q&A

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain,
    )

    return rag_chain

#### 0) Important comment about tokens and template

Just one detail first: this part of the code below was created only to adapt the prompt **if** we are using the hugging face pipeline like HuggingFaceHub, because remember that at the current time this LangChain implementation does not always correctly identify the stop tokens for certain models.

If you don't remember, see the 2nd project's Colab.

Here we are using Llama 3 so the start tokens (which we defined as `token_s`) and end tokens (`token_e`) are those below, so if you use another model - like Phi 3 - remember to adapt them to the appropriate format. We will concatenate these variables to our prompts

In other words: if you load the LLM through the hugging face hub pipeline then it applies the tokens; otherwise, it is not necessary, so these two variables will have no text (which means that the prompt will remain the same and unchanged).

model_class = "hf_your_huggingface_key_here"
# ...

if model_class.startswith("hf"):
    token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
else:
    token_s, token_e = "", ""

#### 1) Query reformulation to contextualize

First, we need to define a sub-chain that uses previous messages and the last question asked to reformulate the question itself, if it refers to information already mentioned in the history.

* To achieve this, first we define the system contextualization prompt in `context_q_system_prompt` that will tell the model to reformulate the question based on the history

* We chose this prompt because it is well accepted and recommended by the library authors, we just modified it a little. Even if you speak another language, we recommend to leave this prompt in english because it has a chance of working better,that's because - like it or not - this model may be better in this language (although we are using a modern model that works well in other languages), but you could change the language here too and test different prompts to see which one gives the best result.
 * Just to mention, another example of a prompt that could be used: "Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question. This is a conversation with a human. Answer the questions you get based on the knowledge you have. If you don't know the answer, just say that you don't, don't try to make up an answer."

* And we also define the user's prompt. Although it is not essential, here we put "Question: " before {input} because in certain tested models this helped reinforce the LLM's better understanding that the following part is the question and therefore this is the part it should reformulate.

* Next, we will use a prompt that includes a variable called "chat_history", which as we have already seen allows us to insert a list of messages in the prompt using the input key "chat_history". These messages will be inserted after the system message and before the user's most recent question.

# Contextualization prompt
context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."

context_q_system_prompt = token_s + context_q_system_prompt
context_q_user_prompt = "Question: {input}" + token_e
context_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", context_q_user_prompt),
    ]
)

* We use the helper function `create_history_aware_retriever` to handle cases where the message history is empty. Otherwise, it applies a sequence of `prompt | llm | StrOutputParser() | retriever`.

* The create_history_aware_retriever function builds a chain that accepts input and chat_history as input and has the same output format as a retriever.

"""# Chain for contextualization
history_aware_retriever = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=context_q_prompt
)"""

In short, this chain adds a reformulation of the input query to our retriever, so that the retrieval incorporates the context of the conversation.

With this, we have everything we need to set up our question and answer chain.

#### 2) Question and answer chain (Q&A)

This is the chain that will return the final answer, which is why it is known by this name in this context.

First, we will create the prompt template, taking the opportunity to change the prompt and adapt it to RAG. We can even use the same template we created for the previous RAG example, but here you can customize it as you wish and add or remove instructions in the prompt. As an addition, we added a sentence asking to return in Portuguese, to reinforce the LLM since part of the content read may be in other languages, so we want it to be translated at the end if necessary.

qa_prompt_template = """You are a helpful virtual assistant answering general questions.
Use the following bits of retrieved context to answer the question.
If you don't know the answer, just say you don't know. Keep your answer concise.
Answer in English. \n\n
Question: {input} \n
Context: {context}"""

qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

LangChain has some functions to facilitate the creation of the chain that we need to do here and that was described above. Instead of using the LCEL syntax seen previously (where each component is separated by `|`), we can use some methods created exactly for this purpose:

* `create_stuff_documents_chain` specifies how the context retrieved (via retriever) is fed into a prompt and LLM. In this case, we will "stuff" the content (hence "stuff" in the function name) in the prompt, that is, we will include all the retrieved context without any summary or other processing.
 * we will use it here to generate the question and answer chain (`qa_chain`), with input keys `context`, `chat_history` and `input`.

* `create_retrieval_chain` - adds the retrieval step and propagates the retrieved context through the chain, providing it together with the final answer.
 * this chain applies history_aware_retriever and qa_chain in sequence, retaining intermediate outputs (such as retrieved context) for convenience. For input: `input` and `chat_history`; and output: `input`, `chat_history`, `context` and `answer`. The final response from the model will be accessible by answer (more on that later)

In other words, this makes it easier to create a document processing chain for Q&A tasks, combining multiple documents into a single context. This function is essential in the context of conversational RAGs, as it allows the model to effectively use the retrieved information together with the user's queries to generate accurate and contextually relevant answers

"""
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain,
)"""

Now that we completed the chain of our advanced RAG, we can return in the function with `return rag_chain`

---

### User Input and Response Generation

We want the model to generate a response only if we upload at least one document. To create it this behavior, we need to modify after `user_query = st.chat_input...`, adding a new condition here: `and uploads is not None`.
after `user_query is not None and user_query != ""`

that is, it will look like this:

"""
if user_query is not None and user_query != "" and uploads is not None:
  ###
"""

#### Retrieving information

Now we need to call the function that executes all the retriever logic.

Let's add this inside `with st.chat_message("AI"):`

Here we will call the config_retriever() function and pass the uploaded files as a parameter (`uploads` variable)

So first we have to get the retrieved information

"""retriever = config_retriever(uploads)"""

#### Pipeline Optimization

We will show you a method we created to optimize the response. If you do it the way above, all the indexing and retrieval steps will be executed for each question asked to the AI.

To make your pipeline more efficient, we can avoid unnecessary re-execution of the process of dividing into chunks, creating embeddings and storing in the vector database every time a new question is asked.

Since the PDF document is not changing between runs, we can configure it to save this list of the processed file names, then we can use it to compare if there was a change - and only re-run the retrieval snippet if there was a change in any document (or if it was removed or added).

First, we will create a variable that stores the list of processed file names and the retriever. This will allow you to keep track of the files already processed between runs of the application.

To make the mode values persist in the current session, we will use Streamlit's st.session_state, which we have previously used to save the chat history. So, right after the if "chat_history", let's add the following:    

if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

After creating our variables that persist in the session, simply adapt the retriever code so that it works like this:

* at each execution, compare the current file list with the list stored in st.session_state. If there are changes (i.e. new files were uploaded or old files were removed), run the chunking and embedding generation process again.

* if the file list has not changed, skip the chunking and embedding generation steps and directly use the retriever stored in the session to search for the information.

To implement this logic, simply change the code, replacing `retriever = config_retriever(uploads)` with the following method:

"""if st.session_state.docs_list != uploads:
    print(uploads)
    st.session_state.docs_list = uploads
    st.session_state.retriever = config_retriever(uploads) """

#### Loading the RAG chain and generating the response

In this first line below, we call the config_rag_chain function, which contains all the logic we defined above. At the end, this chain is assigned to the rag_chain variable.

In the second line, the configured RAG chain is invoked using the invoke method, which receives a dictionary with two keys: input (which contains the user's current query, in this case, user_query), and chat_history (which includes the conversation history stored in st.session_state.chat_history). Remember that the conversation history is used to provide additional context, helping the model generate a more relevant and accurate response based on previous interactions.

The result of this execution is stored in the result variable, which contains the final response generated by the chain.

"""
rag_chain = config_rag_chain(model_class, st.session_state.retriever)

result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
"""

### Displaying the answer

The line below extracts the answer generated by the AI - the `result` variable - which contains the result of the execution of the RAG chain. The value of result is a dictionary and the key 'answer' stores the final answer. Therefore, when accessing result['answer'], we directly obtain the text generated by the LLM in response to the query/question we provided

And to display it in the interface within the chat we use again `st.write`, or `st.markdown`

"""resp = result['answer']
st.write(resp)"""

And again here, the `resp` will also be added to the chat history, with `st.session_state.chat_history.append(AIMessage(content=resp))`

### Bonus: showing the source of the information obtained

This will be useful especially when loading multiple documents

Here we can set to display only the first reference, or all of them. The number of sources here is related to the k parameter of vectorstore.as_retriever

* `sources = result['context']`: This line extracts the list of documents/sources used by the model from the 'context' field of the result returned by LLM. These documents contain the information that helped generate the response.

* `for idx, doc in enumerate(sources):`: a loop that goes through the list of sources documents, enumerating each document. The idx index will be used to number the sources, and doc represents the current document in each iteration.

* `source = doc.metadata['source']`: For each document, the line accesses the value associated with the 'source' key within the document's metadata. This value typically contains the path or URL from which the document was loaded.

* `file = os.path.basename(source)`: This line extracts the base name of the file (i.e. the file name without the full path) from the source variable. This makes the name more readable for display.

* `page = doc.metadata.get('page', 'Page not specified')`: Here, the code attempts to obtain the document's page number from the metadata. If the page is not specified, the default value 'Page not specified' is used.

* `ref = f":link: Source {idx}: *{file} - p. {page}*"`: This line formats a string representing the source reference. The source index, file name, and page number are displayed, resulting in a reference like "Source 1: document.pdf - p. 2".

* `print(ref)`: prints the formatted reference to the console, only for debugging or quick viewing.

* `with st.popover(ref):`: Starts a popover (interactive visual element) in Streamlit. This popover will be triggered when the user interacts with the displayed source reference. We use the popover component because we find it interesting in this situation, if you want others components you can find them here https://docs.streamlit.io/develop/api-reference/layout

* `st.caption(doc.page_content)`: Inside the popover, the page content (extracted from the doc.page_content variable) is displayed as a caption. So, when the user clicks on the popover they will be able to see the page text directly related to the cited source.

"""
sources = result['context']
for idx, doc in enumerate(sources):
    source = doc.metadata['source']
    file = os.path.basename(source)
    page = doc.metadata.get('page', 'Page not specified')

    ref = f":link: Source {idx}: *{file} - p. {page}*"
    print(ref)
    with st.popover(ref):
        st.caption(doc.page_content)
"""

And finally, we will use the time library to count how long the generation took.

start = time.time()
# rest of the code [...]
end = time.time()
print("Time: ", end - start)

---
---

## Lauching the interface

Finally, we gathered all the code into a single script and added the page configuration with st.set_page_config and st.title. Despite other changes in the logic of our application, compared to project 2 we also changed the title and emoji to make the interface more personalized and suitable for the current project, with a look more aligned with the context of this project.

%%writefile proj03.py

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

import torch
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface  import HuggingFaceEndpoint

import faiss
import tempfile
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader


from dotenv import load_dotenv

load_dotenv()

# Streamlit Settings
st.set_page_config(page_title="Chat with documents 📚", page_icon="📚")
st.title("Chat with documents 📚")

model_class = "openai" # @param ["hf_your_huggingface_key_here", "openai", "ollama"]

## Model Providers
def model_hf_your_huggingface_key_here(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
  llm = HuggingFaceEndpoint(
      repo_id=model,
      temperature=temperature,
      max_new_tokens=512,
      return_full_text=False,
      task="text-generation",
      #model_kwargs={
      #    "max_length": 64,
      #    #"stop": ["<|eot_id|>"],
      #}
  )
  return llm

def model_openai(model="gpt-4o", temperature=0.0):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
        # other parameters...
    )
    return llm

def model_ollama(model="phi3", temperature=0.1):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm


## Indexing and Retrieval

def config_retriever(uploads):
    # Load
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

    # Store
    vectorstore = FAISS.from_documents(splits, embeddings)

    vectorstore.save_local('vectorstore/db_faiss')

    # Retrieve
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k':3, 'fetch_k':4}
    )

    return retriever


def config_rag_chain(model_class, retriever):

    ### Loading the LLM
    if model_class == "hf_your_huggingface_key_here":
        llm = model_hf_your_huggingface_key_here()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()

    # Prompt definition
    if model_class.startswith("hf"):
        token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        token_s, token_e = "", ""

    # Contextualization prompt
    context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."

    context_q_system_prompt = token_s + context_q_system_prompt
    context_q_user_prompt = "Question: {input}" + token_e
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", context_q_user_prompt),
        ]
    )

    # Chain for contextualization
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=context_q_prompt
    )

    # Q&A Prompt
    qa_prompt_template = """Sen Türk Ceza Kanunu uzmanı bir hukuk danışmanısın. Kullanıcının sorusunu yalnızca aşağıdaki PDF'deki bilgilerle "
        "yanıtla. Cevabında şu bilgilere özellikle yer ver:\n"
        "- Hangi maddeyle ilgilidir\n"
        "- Suçun tanımı ve cezanın türü (hapis, para cezası, vb)\n"
        "- Ceza süresi veya özel durumlar (örneğin şartlı tahliye, artırıcı nedenler)\n"
        "Cevaplar teknik, açık ve PDF kaynaklı olmalı. PDF'de belirtilmemişse 'Belirtilmemiştir' yaz. "
        "Maksimum 5 cümle kullan.\n\n
    Question: {input} \n
    Context: {context}"""

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

    # Configure LLM and Chain for Q&A

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain,
    )

    return rag_chain


## Creates side panel in the interface
uploads = st.sidebar.file_uploader(
    label="Upload files", type=["pdf"],
    accept_multiple_files=True
)
if not uploads:
    st.info("Please send some file to continue!")
    st.stop()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi, I'm your virtual assistant! How can I help you?"),
    ]

if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# we use time to measure how long it took for generation
start = time.time()
user_query = st.chat_input("Enter your message here...")

if user_query is not None and user_query != "" and uploads is not None:

    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):

        if st.session_state.docs_list != uploads:
            print(uploads)
            st.session_state.docs_list = uploads
            st.session_state.retriever = config_retriever(uploads)

        rag_chain = config_rag_chain(model_class, st.session_state.retriever)





        result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})

        resp = result['answer']
        st.write(resp)

        # show the source
        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page', 'Page not specified')

            ref = f":link: Source {idx}: *{file} - p. {page}*"
            print(ref)
            with st.popover(ref):
                st.caption(doc.page_content)

        # ✅ Now outside the loop (correct!)
        st.session_state.chat_history.append(AIMessage(content=resp))

end = time.time()
print("Time: ", end - start)

### Running Streamlit




!streamlit run proj03.py &>/content/logs.txt &

And now to connect, we use the command below (explanations in colab from project 02)

!wget -q -O - ipv4.icanhazip.com

!npx localtunnel --port 8501

## Testing the Application
The first question will take a bit longer because it needs to run all RAG stages, so the time will vary based on the content size. However, the next question should be much faster, as we’ve programmed the application to avoid unnecessary processing (remember, it will only re-run the indexing and retrieval stages if you add or remove a file).

**(suggestion of what to type)**

1. Upload the file "BlueNexus Industries Presentation.pdf" in the side panel (this and the other PDFs mentioned are located in the documents folder on Drive). This is a presentation we created as an example for a fictional company, which contains various details about the company.
Type the following messages:

* `tell me about BlueNexus`

* `when was it founded?`

* `what was the revenue for the last year?`

* `who is Dr. Watson?`

  * a note: here it shows how the retrieved context holds much more weight than the LLM's own knowledge (mainly because we requested in the prompt to "ignore" and only generate responses based on the retrieved context). This is a good example to notice because "Dr. Watson" is related to a well-known character in pop culture (from "Sherlock Holmes"), and the model likely has knowledge of this. However, as it’s using only the retrieved context, it won’t be confused with the other Dr. Watson — it will correctly respond by talking about the company’s founder.

2. Now, upload the "Attention-is-All-You-Need-Paper.pdf." You’ll notice it takes a bit longer to respond when asking a question since loading a document means we need to re-run all indexing and retrieval stages.
Ask:

* `what is attention?`
  * again, this shows how the AI considers the retrieved context. Here, "attention" refers to the mechanism discussed in the paper, rather than the general meaning of the word "attention."

## How to improve 🚀

By knowing exactly how each function operates and understanding the explanations covered throughout this project, you have the necessary knowledge to improve the results of your RAG application. Below we will list the strategies that can be applied to optimize the quality of the responses and the efficiency of the system:

* Test other Embedding models - as mentioned, selecting the correct model for the RAG system is crucial, as it directly affects the accuracy of the responses, in addition to the use of resources and the scalability of the application. Choosing models that work well with the language and the task in question can significantly improve the results. By testing different models, you can identify which one offers the best combination of quality and efficiency for your needs.

* Adjust the fixed (system) prompt - Modifying the system prompt to make it more explicit about the functions that the LLM should perform can improve the results. The prompt should clearly specify what the LLM should prioritize in the response and what should be ignored. This guides the model to focus on what is most relevant to your application and your goal.

* Improve the user prompt - remind the user (perhaps by placing a warning in the interface) that the more specific the question is, the greater the chance of increasing the accuracy of the answers generated by LLM. The more detailed and clear the request, the more relevant the return will be. This practice also helps to reduce ambiguities that can harm the model's interpretation of the query.

* Adjust the contextualization prompt - remember that this prompt reformulates the user's question based on the conversation history, something useful when the query needs context to be correctly interpreted. The contextualization prompt (context_q_system_prompt) instructs the model to take the history into account.

* Test other LLMs - Exploring other language models, especially those that accept a larger number of tokens and perform well in the chosen language, can improve performance. For more demanding cases, it may be worth considering proprietary solutions such as ChatGPT or paid services (such as Groq, mentioned in Colab 1) that provide large open-source models. Larger models can better handle complex queries and provide more elaborate responses.

* Adjust the retrieval parameters (k and fetch_k) - Modifying the parameters of the retrieval steps, such as the values ​​of k and fetch_k, can have a significant impact on the performance of your application. Try starting with smaller values ​​and increasing them as necessary, always monitoring the impact on the relevance and quality of the responses. For more details, see the section on the RAG pipeline and the retriever. Another idea would be to test other algorithms besides MMR.

* Make it better prepared to accept any document - one idea is to preprocess PDF files (or other formats) to adapt them to the vector store. PDFs often have tables or other structures that make interpretation difficult; or documents in different formats such as HTML, CSV, or PPTX are not structured for optimal information extraction. Preparing these files is crucial to ensure that relevant content is correctly captured and made available to the retrieval system.
* There are specialized solutions that automate this transformation, organizing the data and eliminating unnecessary information. This optimizes the workflow and improves the accuracy of the results. One example is the Unstructured service (Visit https://unstructured.io), which facilitates the extraction of complex data from files, making them ready for use in vector databases and LLM frameworks, which increases the quality of information retrieval and the performance of the RAG application.
* To use this in langchain is simple, you can use the Document Loader method. In practice, just load the document using the Unstructured document loader (instead of the PyPDFLoader that we used). More details here: https://python.langchain.com/v0.2/docs/integrations/document_loaders/unstructured_file/

These strategies aim to optimize the efficiency and quality of the RAG system's responses, adapting it to your specific use case.