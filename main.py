import cv2
import logging
import threading
from semantic_kernel.contents import ChatHistory
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
import configs
from tools import RemoteController
import web_server
from google import genai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    logging.info("Starting the program.")

    

    client = genai.Client(api_key="GOOGLE_API_KEY")

    my_file = client.files.upload(file="path/to/sample.jpg")

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[my_file, "Caption this image."],
    )

    print(response.text)
    
    web_thread = threading.Thread(target=web_server.start_server, daemon=True)
    web_thread.start()
    logging.info("Web server started on http://localhost:5000")

    # Initialize Semantic Kernel and Ollama
    chat_completion = OllamaChatCompletion(
        ai_model_id='llama3.2'
    )
    kernel = Kernel()
    kernel.add_service(chat_completion)

    # Initialize remote controller
    controller = RemoteController(
        bluetooth_port=configs.BLUETOOTH_PORT,
        baud_rate=configs.BAUD_RATE,
    )
    logging.info("Remote controller initialized")

    # Initialize video capture
    cap = cv2.VideoCapture(configs.VIDEO_URL)
    if not cap.isOpened():
        logging.error("Failed to open video capture.")
        return
    logging.info("Video capture started.")

    # Initialize chat history properly
    chat_history = ChatHistory()
    chat_history.add_system_message("Your job is describing image so the toy car can navigate the obstacles.")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("No frame retrieved. Exiting loop.")
                break

            # Update the web interface with the current frame
            web_server.update_frame(frame)

            if frame_count % configs.FRAME_INTERVAL == 0:
                # Convert frame to bytes
                _, buffer = cv2.imencode('.png', frame)
                frame_bytes = buffer.tobytes()
                
                # Create message content with image

                
                # Add the message to chat history
                chat_history.add_message(message)

                # Get AI response
                response = chat_completion.get_chat_message_content(chat_history)
                logging.info(f"AI Response: {response}")
                
                # Update web interface with agent's thoughts
                web_server.update_thoughts(str(response))


            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exit command received. Exiting loop.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Program terminated.")

if __name__ == "__main__":
    main()