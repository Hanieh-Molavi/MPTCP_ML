import os

directory = r"C:/Users/Hanieh/source/packets/3-Fin"
counter = 2934063

for filename in os.listdir(directory):
    if filename.endswith(".pcap") == False:
        old_path = os.path.join(directory, filename)
        
        new_filename = str(counter) + '.pcap'
        new_path = os.path.join(directory, new_filename)
        

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} to {new_filename}")

        counter = counter + 1


directory = r"C:/Users/Hanieh/source/packets/4-Fin"
counter1 = 1

for filename in os.listdir(directory):

    old_path = os.path.join(directory, filename)
    

    new_filename = str(counter1) + '.pcap'
    new_path = os.path.join(directory, new_filename)
    

    os.rename(old_path, new_path)
    print(f"Renamed: {filename} to {new_filename}")

    counter1 = counter1 + 1