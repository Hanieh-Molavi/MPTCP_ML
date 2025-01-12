import pandas as pd


file1 = "C:/Users/Hanieh/source/final/1-x-Else-Fin/1-Fin-features.csv" 


for i in range(6,71):
    file2 = "C:/Users/Hanieh/source/final/1-x-Else-Fin/1-Else-features_"+ str(i) +".csv"  

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    merged_df = pd.concat([df1, df2], axis=0)


    output_file = "1-"+ str(i) +"-Else-Fin.csv" 
    merged_df.to_csv(output_file, index=False)

    print(f"Merged file saved as {output_file}")
