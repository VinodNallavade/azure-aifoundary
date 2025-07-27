# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
import re
import subprocess

from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.functions import KernelFunctionFromPrompt
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define agent names
BUSINESS_ANALYST = "BusinessAnalyst"
SOFTWARE_ENGINEER = "SoftwareEngineer"
PRODUCT_MANAGER = "ProductManager"

chat_completion_service = AzureChatCompletion(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),  # Ensure you set this in your environment
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # Ensure you set this in your environment
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # Ensure you set this in your environment
   api_version= os.getenv("AZURE_OPENAI_API_VERSION"),  # Default to latest version if not set  

)

def create_kernel() -> Kernel:
    """Creates a Kernel instance with an Azure OpenAI ChatCompletion service."""
    kernel = Kernel()
    kernel.add_service(service=chat_completion_service)
    return kernel


def extract_html_from_chat_history(chat_history: list[str]) -> str | None:
    """
    Extracts the HTML code block from the chat history.
    Looks for content between ```html and ```
    """
    pattern = r"```html\n(.*?)\n```"
    for message in reversed(chat_history):
        match = re.search(pattern, message, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None

def save_html_to_file(html_code: str, filename: str = "index.html") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_code)
    print(f"✅ HTML saved to {filename}")



def push_to_github():
    try:
        result = subprocess.Popen(["push_to_github.sh", "Auto-update index.html"], shell=True)
        print("Git push succeeded:")
    except subprocess.CalledProcessError as e:
        print(" Git push failed")


async def main():
    # Create a single kernel instance for all agents.
    kernel = create_kernel()

    # Create ChatCompletionAgents using the same kernel.
    # Create agents for different roles
    code_agent = ChatCompletionAgent(
        name=SOFTWARE_ENGINEER, 
        kernel=kernel,
        instructions= """
        You are a professional Software Engineer. Your task is to write efficient, well-structured HTML, CSS, and JS code based on clear requirements.
        All code should be in one code block, and you should not include any explanations or comments.
        Output clean code only, enclosed within appropriate tags. Do not include explanations unless explicitly asked.   
        """
        ) 
    business_analyst_agent = ChatCompletionAgent(
        name=BUSINESS_ANALYST, 
        kernel=kernel,
        instructions="""
        You are a skilled Business Analyst. Based on the given instructions, your task is to view the requirements and create 
        a clear and concise summary of the requirements.
        Your summary should be structured, highlighting key points and ensuring clarity for next agent.
        Do not include any code or technical details, just a summary of the requirements.
        """

        )
    product_manager_agent = ChatCompletionAgent(
        name=PRODUCT_MANAGER, 
        kernel=kernel,
        instructions="""
         You are the Product Owner. Your job is to ensure that the Software Engineer's code meets all of the user's original requirements.

    Your responsibilities:
    - Cross-check the HTML implementation against the Business Analyst’s requirements.
    - Ensure all features are present and functioning as described. If yes please acknowledge with "READY FOR USER APPROVAL"
    - If any requirements are missing or not implemented correctly, provide specific feedback on what needs to be fixed.
    - Confirm that the Software Engineer has shared the code using this required format:
       ```html
       <!-- HTML content -->
        """

        )




    selection_function = KernelFunctionFromPrompt(
        function_name="selection", 
        prompt=f"""
    Examine the provided RESPONSE and choose the next participant.
    State only the name of the chosen participant without explanation.
    Never choose the participant named in the RESPONSE.

    Choose only from these participants:
    - {BUSINESS_ANALYST}
    - {SOFTWARE_ENGINEER}
    - {PRODUCT_MANAGER}

    Rules:
    - If RESPONSE is user input, it is {BUSINESS_ANALYST} turn.
    - If RESPONSE is by {BUSINESS_ANALYST}, it is {SOFTWARE_ENGINEER} turn.
    - If RESPONSE is by {SOFTWARE_ENGINEER}, it is {PRODUCT_MANAGER} turn.
    - If RESPONSE is by {PRODUCT_MANAGER} and it does not contain 'READY FOR USER APPROVAL' And requirements are not clear then, it is {BUSINESS_ANALYST} turn.
    - If RESPONSE is by {PRODUCT_MANAGER} and it does not contain 'READY FOR USER APPROVAL', it is {SOFTWARE_ENGINEER} turn.
    - If RESPONSE is by {PRODUCT_MANAGER} and it contains 'READY FOR USER APPROVAL', , select {PRODUCT_MANAGER} again (the termination strategy will stop the loop).

    RESPONSE:
    {{{{$lastmessage}}}}
    """
    )

    termination_keyword = "READY FOR USER APPROVAL"

    termination_function = KernelFunctionFromPrompt(
            function_name="termination",
            prompt=f"""
        You are evaluating whether the conversation should end.

        Rules:
        - If the RESPONSE contains the word "{termination_keyword}" (case-insensitive), respond with a single word: "{termination_keyword}".
        - Otherwise, respond with "CONTINUE".

        Only respond with one word. Do not explain.

        RESPONSE:
        {{{{$lastmessage}}}}
        """
        )

    history_reducer = ChatHistoryTruncationReducer(target_count=1)

    chat = AgentGroupChat(
        agents=[business_analyst_agent, code_agent, product_manager_agent],
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=business_analyst_agent,
            function=selection_function,
            kernel=kernel,
            result_parser=lambda result: str(result.value[0]).strip() if result.value[0] is not None else SOFTWARE_ENGINEER,
            history_variable_name="lastmessage",
            history_reducer=history_reducer,
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[product_manager_agent],
            function=termination_function,
            kernel=kernel,
            result_parser=lambda result: termination_keyword in str(result.value[0]).upper(),
            history_variable_name="lastmessage",
            maximum_iterations=10,
            history_reducer=history_reducer,
        ),
    )

    is_complete = False
    while not is_complete:
        print()
        user_input = input("User > ").strip()
        if not user_input:
            continue
        
        if user_input.lower() == "exit":
            is_complete = True
            break
        
        if user_input.lower() == "reset":
            await chat.reset()
            print("[Conversation has been reset]")
            continue
        
        # Try to grab files from the script's current directory
        if len(user_input) > 1:
            await chat.add_chat_message(message=user_input)

        try:
            async for response in chat.invoke():
                if response is None or not response.name:
                    continue
                print()
                print(f"# {response.name.upper()}:\n{response.content}")
        except Exception as e:
            print(f"Error during chat invocation: {e}")
        print(chat.is_complete)
        if chat.is_complete:
            print("***********************[Conversation complete]************************")
            print("Conversation ended with APPROVED.")
    
            # Gather full chat history as plain strings
            messages = [msg.content for msg in chat.history]

            # Try to extract and save the HTML
            html_code = extract_html_from_chat_history(messages)
            if html_code:
                save_html_to_file(html_code)
                push_to_github()
            else:
                print("No valid HTML block found in chat history.")

        # Reset the chat's complete flag for the new conversation round.
        chat.is_complete = False


if __name__ == "__main__":
    asyncio.run(main())