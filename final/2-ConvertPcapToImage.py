from scapy.all import rdpcap, wrpcap, Ether, IP
import numpy as np
from PIL import Image


def pcap_to_image(pcap_files, image_file):

    data = b''
    for pcap_file in pcap_files:

        packets = rdpcap('C:/Users/Hanieh/source/final/packets/4-Fin/' + pcap_file)
        data += b''.join(bytes(packet) for packet in packets)

    data = data[:50*50*3]
    data = data.ljust(50*50*3, b'\0') 

    img_data = np.frombuffer(data, dtype=np.uint8)
    img_data = img_data.reshape((50, 50, 3))

    img = Image.fromarray(img_data)
    img.save(image_file, 'JPEG')

for i in range(1,42054, 50):
    pcap_files = ["C:/Users/Hanieh/source/final/packets/4-Fin/" + str(j) + ".pcap" for j in range(i, i+50)]

    output_pcap_files =  [str(j) + '.pcap' for j in range(i, i+50)]
    counter = (i//50)
    
    image_file =  str(counter) + '.jpeg'
    pcap_to_image(output_pcap_files, image_file)
