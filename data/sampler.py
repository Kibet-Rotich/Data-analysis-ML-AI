import pandas as pd

def sample_emails(input_file, output_file, sample_size=10000, random_state=42):
    """
    Sample random records from a CSV file and save them to a new CSV.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path where the sampled dataset will be saved
        sample_size (int): Number of records to sample
        random_state (int): Random seed for reproducibility
    """
    # Read the original dataset
    print("Reading the original dataset...")
    df = pd.read_csv(input_file)
    
    # Take random sample
    print(f"Sampling {sample_size} records from {len(df)} total records...")
    sampled_df = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    
    # Save to new file
    sampled_df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(sampled_df)} records to {output_file}")
    
    return sampled_df

# Example usage
if __name__ == "__main__":
    INPUT_FILE = "emails.csv"
    OUTPUT_FILE = "sampled_enron_emails.csv"
    
    df = sample_emails(INPUT_FILE, OUTPUT_FILE)
    print(f"Dataset shape: {df.shape}")