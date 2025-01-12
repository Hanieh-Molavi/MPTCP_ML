from scapy.all import rdpcap , wrpcap
import json

def extract_streams(pcap_file):
    packets = rdpcap(pcap_file)
    
    for y in range(0,len(pcap_file)):

        if 'IP' in packets[y]:     
            packets[y]['IP'].src = "0.0.0.0"
            packets[y]['IP'].dst = "0.0.0.0"

        if "TCP" in packets[y]:
            z = y + 20
            if packets[y]["TCP"].flags.value & 0x01:
                wrpcap(f"C:/Users/Hanieh/source/final/Fin/" + str(z),packets[y])

            else:
                wrpcap(f"C:/Users/Hanieh/source/final/Else/" + str(z),packets[y] )


# for i in range(1,296):
pcap_file = r"C:/Users/Hanieh/source/final/mptcp-.pcap"
streams = extract_streams(pcap_file)

