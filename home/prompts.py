from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, AgentExecutor



def manual_prompt(query, message_history, db, llm):
    
    manual_prompt = ChatPromptTemplate.from_template('''
        you are a helpful reaserch paper assistant 

        you are given the context of the reaserch paper from the Faiss db. Answer the user based on the context 
                                                
        try to give maximum information on the context based on the users input

        ````````
        Example:

        (Context:
        Digital twins are increasingly used in manufacturing to optimize production processes. By creating a virtual model of a physical system, manufacturers can simulate and analyze performance, predict failures, and implement improvements without interrupting operations. Case studies show significant cost savings and efficiency gains.

        Input:
        "How do digital twins benefit manufacturers?"

        Answer:
        Digital twins benefit manufacturers by allowing them to simulate and analyze performance, predict failures, and implement improvements without interrupting operations, leading to significant cost savings and efficiency gains.) -> example

        Begin! 
        ````````

        Context: 
        {context}

        Input: 
        {input}

        Chat History:
        {chat_history}

        Answer:
    ''')
    
    output_parser = StrOutputParser()
    chain = manual_prompt | llm | output_parser
    result = db.similarity_search(query)
    similar_docs = '\n\n'.join([doc.page_content for doc in result])
    res = chain.invoke({'context': similar_docs, 'input': query, 'chat_history': message_history})
    return res

def react_prompt(query, message_history, db, llm):
    template = ChatPromptTemplate.from_template('''
        Assistant is a large language model trained by Gemini.

        Assistant is designed to be able to assist with answering the questions from the research paper, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
                                                    
        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

                                                    
        Note: always use one of the tools                                             

        TOOLS:
        ------

        Assistant has access to the following tools to get relevant results from the reaserch paper:

        {tools}

        To use a tool, please use the following format:

        ```
        Thought: Do I need to use the tool? Yes
        Action: the action to take, should be in [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ```
        ----- Repeat N times 

        When you have a response to say to the Human you MUST use the format:

        ```
        Final Answer: [your response here]
        ```

        Begin!

        Previous conversation history:
        {chat_history}

        New input: {input}
        {agent_scratchpad}
        ''')
    
    retriever = db.as_retriever()
    
    retriever_tool = create_retriever_tool(
        retriever,
        "context_retriever",
        "Searches and returns the context that are in the reaserch_paper(docs) which contains information on what the user is speaking about",
    )

    tools = [retriever_tool]

    agent = create_react_agent(llm, tools=tools, prompt=template)
    agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)
    result = agent_executer.invoke({'input': query, 'chat_history': message_history})
    return result['output']


def normal_rag(query, message_history, db, llm):
    template = '''
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Chat_History: {chat_history}
        Answer:
        '''
    output_parser = StrOutputParser()
     
    chain = template | llm | output_parser

    result = db.similarity_search(query)

    similar_docs = '\n\n'.join([doc.page_content for doc in result])

    output_parser = StrOutputParser()

    res = chain.invoke({'context': similar_docs, 'input': query, 'chat_history': message_history})
    return res

# template = ChatPromptTemplate.from_messages([('system',
#     '''
#     You are a chat assitant for helping the user to find the reaserch paper 

#     Only ask three question one at a time.
    
#     After asking three questions (only one at a time), create a query to search in database

    
#     Example:
#     `````
#     Input: Sentiment analysis usig transformers network
#     Answer: Is there any unique network you are interested in?
#     Input: no i dont care about the transformer architecture as long as it is good for getting accurate res 
#     Answer: Is there anything else you want to hadd wich help me find the best paper for you
#     Input: No, I dont have any other points
#     (transformer should be related to deeplearning, he doesnt care about the architecture so any architecture with high acc for predicting the sentiment of the task)
#     Answer: Query: The transformer network in sentiment analysis tasks which comes with good performance
#     `````
#     Begin! 
#     '''), 
#     MessagesPlaceholder('messages')
#     ])

template_t1 = ChatPromptTemplate.from_messages([('system',
    '''
    You are a chat assitant for helping the user to find the reaserch paper 
    After asking questions or having a conversation (only one at a time), create a query which searches for the paper in the database.
    when user ask for research paper on quantumn mechanics dont include term research paper in the query, the query must be precise on the topic that research paper talks about. 
    You should never reveal the (thought: ....) to the user only reply with the Answer only this is really important

    (The examples shows the conversational history, should only answer one at a time and wait for the user input)
    Examples:

    Example 1:
    ```
    Sentiment analysis using transformers network (I need more information the paper to search for)
    Answer: Is there any unique network you are interested in?
    no i dont care about the transformer architecture as long as it is good for getting accurate result (he cares about the accuracy and can be any architecture so no need to remeber architecture)
    Answer: Is there anything else you want to add which help me find the best paper for you 
    No, I dont have any other points (transformer should be related to deeplearning, he doesnt care about the architecture so any architecture with high acc for predicting the sentiment of the task should search paper by creating Query)
    Answer: Query: The transformer network in sentiment analysis tasks which comes with good performance
    ```

    Example 2:
    ```
    hello what is this (The user is new and doesnot know about the app)
    Anwer: Hi, Im Scholar Compass an ai assistance for getting you the research paper that you need. Is there any research paper you are looking for?
    No, Im just here to get some advice (he is not here to get the research paper so no need to create query lets end the conversation)
    Answer: Im sorry, but i am created in order to help assist you to get research paper, as such im unable to help you, Thank you and have a nice day.  
    ```

    ```
    i need a research paper on Quantumn mecahincs (what specifics is he looking for in this paper)
    Answer: there are lots of papers in Quantumn mechanics may i know your research specifics?
    im just looking for basic quantumn mechanics paper. (he just needs a basic quantumn mechanics paper. should create the search Query)
    Answer: Query: Basics of Quantumn mechanics 
    Nevermind i need a paper on quantumn mechanics which talks about space in the paper (he must have seen the basic quantumn mechanics paper. now he want the quantumn mechanic paper which talks about space. some suggestions might help him get specific results)
    Answer: Do you any specific research papers on these topics related to space: Quantum Gravity, Quantum Field Theory in Curved Spacetime, ... 
    No, i want to learn about quantumn mechanics workings in space (should search for applications of quantumn mechanics in space)
    Answer: Query: Applications of quantum mechanics in space
    ```

    ```
    I need a research paper on biology (i should make sure to get him papers based on his research)
    Answer: Give me your research topic or the issue that you are facing i might be helpful to suggest you some options
    Im facing problems when im looking through the microscope (should help this person with his issue find his problem and suggest a topic for search which helps him with his issue)
    Answer: You can look for the research paper which talks about the use of microscopy in pharmacology and cell biology research in research environment or something else based on your exact issue. what is your exact issue. i think i might be of help to you.
    .......
    Answer: Query: Microscopy techniques in molecular biology
    ```

    ```
    I need a research paper on human anatomy (i should suggest some topics and ask whether is it what he/she is looking for)
    Answer: Do you need research paper specific to any topics lik The Musculoskeletal System, Cardiovascular System and Heart Diseases etc??
    ......
    Make a query with bold format and with some description (The query format should be strictly 'Query: ...' bold format or any other format is not allowed)
    Answer: Query: The Musculoskeletal System
    Begin! 
    '''), 
    MessagesPlaceholder('messages')
    ])

