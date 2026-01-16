import pandas as pd
import os
from pathlib import Path

# Directory principale contenente i gruppi
csv_train_dir = "CSV_train"

# Itera attraverso tutti i gruppi (group0 - group8)
for group_num in range(9):
    group_name = f"group{group_num}"
    group_path = os.path.join(csv_train_dir, group_name)
    print(Path.cwd())
    print(group_path)
    
    # Verifica se la directory del gruppo esiste
    if not os.path.exists(group_path):
        print(f"Directory {group_name} non trovata, skip...")
        continue
    
    # Trova tutti i file CSV nel gruppo (escludendo file già combinati)
    csv_files = [f for f in os.listdir(group_path) 
                 if f.endswith('.csv') and f.startswith('dataset_user_')]
    
    if not csv_files:
        print(f"{group_name}: nessun file CSV da unire")
        continue
    
    print(f"\n{group_name}: trovati {len(csv_files)} file CSV")
    
    dfs = []
    for file in csv_files:
        file_path = os.path.join(group_path, file)
        # Leggi il file CSV specificando il delimitatore ';'
        df = pd.read_csv(file_path, sep=';')
        
        # Rimuovi la colonna 'Unnamed: 0' se presente
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            
        dfs.append(df)
    
    # Concatena tutti i DataFrame in uno solo
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Unisci le colonne di respirazione (prendi il valore non-null da entrambe)
    if 'resp_avgSleepRespirationValue' in combined_df.columns and 'sleep_averageRespirationValue' in combined_df.columns:
        combined_df['resp_avgSleepRespirationValue'] = combined_df['resp_avgSleepRespirationValue'].fillna(
            combined_df['sleep_averageRespirationValue']
        )
        combined_df = combined_df.drop(columns=['sleep_averageRespirationValue'])
    
    # Rimuovi le colonne 'day' e 'act_activeTime' se presenti
    columns_to_drop = ['day', 'act_activeTime']
    columns_to_drop = [col for col in columns_to_drop if col in combined_df.columns]
    if columns_to_drop:
        combined_df = combined_df.drop(columns=columns_to_drop)
    
    # Salva il risultato in un nuovo file CSV nella directory del gruppo
    output_filename = os.path.join(group_path, f"{group_name}_combined.csv")
    combined_df.to_csv(output_filename, index=False, sep=';')
    
    print(f"  → Salvato: {output_filename}")
    print(f"  → Dimensioni: {combined_df.shape}")

print("\n✓ Completato!")