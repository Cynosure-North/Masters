import natnet
import pprint
import time

frames = []

def receive_new_frame(frame):
    frames.append(frame)

def receive_new_desc(desc):
    frames[0] = desc

streaming_client = natnet.NatNetClient(server_ip_address="127.0.0.1", local_ip_address="127.0.0.1", use_multicast=False)
streaming_client.on_data_frame_received_event.handlers.append(receive_new_frame)
streaming_client.on_data_description_received_event.handlers.append(receive_new_desc)

with streaming_client:
    streaming_client.request_modeldef()
    streaming_client.update_sync()
 
    time.sleep(10)
    print("starting recording")

    for i in range (5):
        time.sleep(1)
    
        streaming_client.update_sync()
        
print("ended recording")
        
with open("out.txt", 'w') as file:
    pprint.pprint(frames, stream=file, width=100, indent=4)