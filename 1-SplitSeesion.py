from scapy.all import rdpcap , wrpcap
import json

def extract_streams(pcap_file):
    packets = rdpcap(pcap_file)
    
    for y in range(0,len(pcap_file)):
        if 'IP' in packets[y]:   

            packets[y]['IP'].src = "0.0.0.0"
            packets[y]['IP'].dst = "0.0.0.0"

        if "MPTCP" in packets[y]["TCP"].options:
            mptcp_options = packets[y]["TCP"].options["MPTCP"]

            if mptcp_options.flags & 0x01:
                wrpcap(f"C:/Users/Hanieh/source/final/Fin/" ,packets[y])

            else:
                wrpcap(f"C:/Users/Hanieh/source/final/Else/" ,packets[y] )

pcap_file = r"C:/Users/Hanieh/source/final/mptcp.pcap"
streams = extract_streams(pcap_file)

