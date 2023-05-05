import socket
import threading
import base64
import select
import io
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import struct
import time
import traceback


class ServerConnection:
    def __init__(self):
        self.__cryptKeyFile = "keyFile.txt"
        self.__serverInfo = ("", 5984) # Replace with the IP address of the server
        self.__ModelInfo = ("../poisonous_plant_classifier_model", "modelFile")
        self.__AIModel = tf.keras.models.load_model(filepath=self.GetSavedModelPath())
        self.IMAGE_SAVE_DIR = "received_images"
        self.__buffSize = 4096
        self.__numQueueClients = 12
        self.__accuracyThreshold = 70.00
        self.__timeoutSeconds = 100
        self.__defMsgSize = 16
        self.__classOptions = ['Atlantic_Poison_Oak', 'Eastern_Poison_Ivy', 'Not', 'Poison_Sumac']

    def handle_connection(self, conn, addr):
        print(f'Connection from {addr}')
    
        try:
            while True:
                # Read the prefix byte
                prefix = conn.recv(1)
                if not prefix:
                    break
    
                # Read the image data
                image_data = b''
                while True:
                    chunk = conn.recv(self.__buffSize)
                    if not chunk:
                        break
                    image_data += chunk
                    if image_data.endswith(b'\xff\xd9'):
                        break
    
                # Save the received image
                file_name = os.path.join(self.IMAGE_SAVE_DIR, f'image_{addr[0]}_{addr[1]}_{time.time()}.jpg')
                self.save_received_image(image_data, file_name)
                print(f'Image saved as {file_name}')
    
                # Process the image and send the results back to the client
                score, predictedPlant = self.ProcessImage(file_name)
                if score >= self.__accuracyThreshold:
                    message = f"10,{score},{predictedPlant}".encode()
                else:
                    message = b"01"
                conn.sendall(message)
                print(f'Response sent to {addr}')
                break
    
        except Exception as e:
            print(f'Exception occurred: {e}')
            traceback.print_exc()
    
        finally:
            conn.close()
            print(f'Connection with {addr} closed')






    def main(self):
        os.makedirs(self.IMAGE_SAVE_DIR, exist_ok=True)

        SERVER_IP, SERVER_PORT = self.GetServerInfo()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((SERVER_IP, SERVER_PORT))
            server_socket.listen()

            print(f'Server listening on {SERVER_IP}:{SERVER_PORT}')

            while True:
                conn, addr = server_socket.accept()
                self.handle_connection(conn, addr)

    def ProcessImage(self, img_name):
        # list of classes
        classList = self.GetClassOptList()
    
        # image processing
        img = tf.keras.utils.load_img(img_name, target_size=(224, 224))
        img_tensor = tf.keras.utils.img_to_array(img)                   
        img_tensor = tf.expand_dims(img_tensor, axis=0)         
        img_tensor /= 255.
    
        prediction = self.__AIModel.predict(img_tensor)
        predicted_class_index = np.argmax(prediction)
    
        score = np.max(prediction) * 100
        predictedPlant = classList[predicted_class_index]
        
        # Print the values for debugging and clarity
        print(f"Image array shape: {img_tensor.shape}")  # Modify this line
        print(f"Predictions shape: {prediction.shape}")
        print(f"Confidence: {score}")
        print(f"Predicted plant: {predictedPlant}")
    
        return (score, predictedPlant)




    def save_received_image(self, image_data, filename="received_image.jpg"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as image_file:
            image_file.write(image_data)

    # Getters for settings
    def GetServerInfo(self):
        return self.__serverInfo

    def GetSavedModelPath(self):
        return self.__ModelInfo[0]
    
    def GetClientQueueSize(self):
        return self.__numQueueClients
    
    def GetBuffSize(self):
        return self.__buffSize
    
    def GetAccThresh(self):
        return self.__accuracyThreshold
    
    def GetTimeoutDelay(self):
        return self.__timeoutSeconds
    
    def GetDefMsgSize(self):
        return self.__defMsgSize
    
    def GetClassOptList(self):
        return self.__classOptions
        
if __name__ == "__main__":
    ServerConnection().main()
    

