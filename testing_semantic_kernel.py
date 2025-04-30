import asyncio
import os
from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents import ChatMessageContent, ImageContent, TextContent
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
import logging
from plugin import LightsPlugin

async def main():
    # Initialize the kernel
    kernel = Kernel()

    chat_completion = OllamaChatCompletion(
        ai_model_id='llama3.2'
    )
    kernel.add_service(chat_completion)

    # Set the logging level for  semantic_kernel.kernel to DEBUG.
    setup_logging()
    logging.getLogger("kernel").setLevel(logging.DEBUG)

    # Add a plugin (the LightsPlugin class is defined below)
    
    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Create a history of the conversation
    history = ChatHistory()
    history.add_system_message("I can help you with both text conversations and analyzing images. You can send me text or share image files.")

    # Initiate a back-and-forth chat
    while True:
        print("\nOptions:")
        print("1. Send text message")
        print("2. Upload image")
        print("3. Exit")
        
        choice = input("Choose an option (1-3): ")

        if choice == "3":
            break

        if choice == "1":
            user_text = input("User > ")
            history.add_message(
                ChatMessageContent(
                    role="user",
                    items=[TextContent(text=user_text)]
                )
            )

        elif choice == "2":
            image_path = input("Enter image path: ")
            if os.path.exists(image_path):
                prompt = input("Enter prompt about the image (optional): ")
                message_items = [ImageContent.from_image_file(path=image_path)]
                if prompt:
                    message_items.insert(0, TextContent(text=prompt))
                
                history.add_message(
                    ChatMessageContent(
                        role="user",
                        items=message_items
                    )
                )
            else:
                print("Image file not found!")
                continue

        # Get the response from the AI
        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
        )

        # Print the results
        print("Assistant > " + str(result))

        # Add the message from the agent to the chat history
        history.add_message(result)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())