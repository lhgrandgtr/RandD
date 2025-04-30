import asyncio
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
import configs as c
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion

from semantic_kernel.connectors.ai.google.google_ai.google_ai_prompt_execution_settings import (
    GoogleAIPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory, ChatMessageContent, ImageContent, TextContent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.functions import KernelArguments

import logging
import threading
import cv2
import datetime

import web_server
from tools import RemoteController

async def main():
    kernel = Kernel()
    history = ChatHistory()
    history.add_system_message("Your job is describing image so the toy car can navigate the obstacles.")

    service_id = "agent"
    chat_completion= GoogleAIChatCompletion(
    gemini_model_id="gemini-2.0-flash",
    api_key=c.GOOGLE_API_KEY,
    service_id=service_id,
    )

    kernel.add_service(chat_completion)

    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Required()

    controller = RemoteController(
        bluetooth_port=c.BLUETOOTH_PORT,
        baud_rate=c.BAUD_RATE,
    )

    kernel.add_plugin(controller, plugin_name="RemoteController", description="Plugin to control the car via Bluetooth")

    web_thread = threading.Thread(target=web_server.start_server, daemon=True)
    web_thread.start()
    logging.info(f"Web server started on {c.SERVER_URL}")

    agent = ChatCompletionAgent(
        kernel=kernel,
        name="DriverAgent",
        instructions=f"""
            You are a helpful assistant that can control a toy car.
            You can anayze images and provide commands to the car.
            You can move the car forward, backward, left, or right 
            """,
        arguments=KernelArguments(settings=settings),
    )

    cap = cv2.VideoCapture(c.VIDEO_URL)
    if not cap.isOpened():
        logging.error("Failed to open video capture.")
        return
    logging.info("Video capture started.")

    frame_count = 0

    thread: ChatHistoryAgentThread = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("No frame retrieved. Exiting loop.")
                break

            if frame_count % c.FRAME_INTERVAL == 0:
                try:
                    saved = cv2.imwrite(c.FRAME_PATH, frame)
                    if not saved:
                        logging.error("Failed to save frame")
                        continue

                    web_server.update_frame(frame)
                    history.add_message(
                        ChatMessageContent(
                            role="user",
                            items=[
                                TextContent(text="Please describe the current scene and any obstacles with directions relative to the toy car."),
                                ImageContent.from_image_file(path=c.FRAME_PATH),
                            ]
                        )
                    )

                    response = agent.invoke(history=history, thread=thread)
                    # response = await chat_completion.get_chat_message_content(chat_history=history, settings=settings)
                    logging.info(f"AI Response: {str(response)}")
                    # async for response in agent.invoke(history=history, thread=thread,):
                    #     print(f"{response.content}")
                    #     web_server.update_thoughts(str(response.content))
                    #     thread = response.thread

                except Exception as e:
                    logging.error(f"Error processing frame: {e}")

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exit command received. Exiting loop.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Program terminated.")


if __name__ == "__main__":
    asyncio.run(main())