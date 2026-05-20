import dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from .prompts import BlossomPrompts
from .state import ChatBotState, ShouldPlaySequence, SequenceSelectorOutput
from .tools import get_available_sequences, play_sequence, speak


dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", verbose=False)

def create_chat_agent(speech: bool = True):
    # Main chatbot prompt
    main_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BlossomPrompts.CHATBOT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Sequence selector prompt
    selector_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BlossomPrompts.SEQUENCE_SELECTOR_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Available sequences:\n{sequences}\n\nBased on the conversation, which ONE sequence should I play? Reply with ONLY the sequence name, nothing else."),
        ]
    )

    should_play_sequence_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BlossomPrompts.SHOULD_PLAY_SEQUENCE_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Based on the conversation, should I play a sequence? Reply with ONLY True or False, nothing else."),
        ]
    )

    main_chain = main_prompt | llm
    selector_chain = selector_prompt | llm.with_structured_output(SequenceSelectorOutput)
    should_play_sequence_chain = should_play_sequence_prompt | llm.with_structured_output(ShouldPlaySequence)

    def chatbot_node(state: ChatBotState):
        """Generate empathetic text response"""
        response = main_chain.invoke({"messages": state.messages})
        return {"messages": [response]}

    def should_play_sequence(state: ChatBotState):
        """Decide if we should play a sequence based on user message emotion"""
        
        output = should_play_sequence_chain.invoke({
            "messages": state.messages
        })

        should_play_sequence = output.should_play_sequence

        return "select_sequence" if should_play_sequence else "speech"

    def select_sequence_node(state: ChatBotState):
        """AI selects appropriate sequence based on conversation emotion"""
        
        sequences_str = get_available_sequences()
        
        output = selector_chain.invoke({
            "messages": state.messages,
            "sequences": sequences_str
        })
        
        sequence_name = output.sequence_name
        
        return {"sequence_to_play": sequence_name}

    def play_sequence_node(state: ChatBotState):
        """Play the selected sequence"""
        sequence_name = state.sequence_to_play
        
        if sequence_name:
            play_sequence(sequence_name)
        
        return {}

    def speech_node(state: ChatBotState):
        response_message = state.messages[-1].content

        if speech:
            speak(response_message)

        return {}

    graph = StateGraph(ChatBotState)

    graph.add_node("chatbot", chatbot_node)
    graph.add_node("select_sequence", select_sequence_node)
    graph.add_node("play_sequence", play_sequence_node)
    graph.add_node("speech", speech_node)

    graph.add_edge(START, "chatbot")
    
    graph.add_conditional_edges(
        "chatbot",
        should_play_sequence,
        {
            "select_sequence": "select_sequence",
            "speech": "speech"
        }
    )

    graph.add_edge("select_sequence", "play_sequence")
    graph.add_edge("select_sequence", "speech")
    
    graph.add_edge("play_sequence", END)
    graph.add_edge("speech", END)
    
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
