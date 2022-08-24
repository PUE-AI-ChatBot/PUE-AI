import socket
from threading import Thread
from models.bertmodels.aimodel import AIModel
import struct
import json

class ServerSession:
    def __init__(self, converters):
        self.__LocalIP = ''
        self.__Port = 9999
        self.__Size = 1024
        self.__Server_Address = (self.__LocalIP, self.__Port)
        self.struct_Lsize = struct.calcsize("L")
        self.__dialogs_buffer = []
        self.__usersList = []
        self.__AddressBook = {}
        self.main_model = AIModel()

    def __answerThread(self, s_socket, pw):
        while True:
            try:
                if len(self.__dialogs_buffer) > 0:
                    c_socket, name, message = self.__dialogs_buffer[0]
                    if name in self.__usersList:
                        print("<{}> loaded one dialog : {}" .format(name, message))
                        dialog_data = self.main_model.run(name, message)
                        print(json.dumps(dialog_data, ensure_ascii=False, indent="\t"))
            
                        c_socket.sendall(dialog_data["System_Corpus"].encode())

                    del self.__dialogs_buffer[0]
                    # print(self.__usersList)
                    # print(self.__AddressBook)
            except Exception as error:
                if error is ConnectionResetError or "[WinError 10038] 소켓 이외의 개체에 작업을 시도했습니다":
                    print("%s client disconntects with server" % (name))
                    if name in self.__usersList: self.__usersList.remove(name)
                else:
                    print("{} : Error occured" .format(error))
                # print(self.__usersList)
                # print(self.__AddressBook)

    def __make_new_chatThread(self, c_socket, c_address, is_newbie):
        while True:
            try:
                received = c_socket.recv(self.__Size)
                nameLength_byte = received[:self.struct_Lsize]
                received_data = received[8:]
                nameLength = struct.unpack("L", nameLength_byte)[0]
                name = received_data[:nameLength].decode()
                data = received_data[nameLength:].decode()
                if is_newbie == True:
                    self.__AddressBook[c_address] = name
                    self.__usersList.append(self.__AddressBook[c_address])
                    is_newbie = False
                if data == 'end':
                    break;
                self.__dialogs_buffer.append((c_socket, name, data))
                print("received by <{}> : {}" .format(name, data))
            except Exception as error:
                if error is ConnectionResetError or "unpack requires a buffer of 4 bytes":
                    print("%s client disconntects with server"%(name))
                    if c_address in self.__usersList : self.__usersList.remove(name)
                else:
                    print("{} : Error occured" .format(error))
                break

        c_socket.close()

    def startSession(self):
        print("Session has started")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(self.__Server_Address)
        server_socket.listen()
        print("listening ..")

        receiving_thread = Thread(target=self.__answerThread, args=(server_socket, 100))
        receiving_thread.daemon = True
        receiving_thread.start()

        while True:
            try:
                client_socket, client_address = server_socket.accept()
                print("Client is connected")
                is_newbie = True

                if client_address in self.__AddressBook: 
                    self.__usersList.append(self.__AddressBook[client_address])
                    is_newbie = False

                user_thread = Thread(target=self.__make_new_chatThread, \
                    args=(client_socket, client_address, is_newbie))
                user_thread.daemon = True
                user_thread.start()
                
            except Exception as error:
                print("Error : {}" .format(error))
                server_socket.close()
                break

    
