import uuid
from langchain_core.messages import AIMessage, HumanMessage
from .chatbot.agent import create_chat_agent


def main():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    chatbot = create_chat_agent()

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        
        print("Blossom: ", end="", flush=True)
        
        ai_response = ""
        for chunk in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]}, 
            config=config,
            stream_mode="updates"
        ):
            if "chatbot" in chunk:
                messages = chunk["chatbot"].get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        ai_response = msg.content
                        print(ai_response, end="\n", flush=True)
        
        print()


if __name__ == "__main__":
    main()