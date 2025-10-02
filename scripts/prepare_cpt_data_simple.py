#!/usr/bin/env python3
"""
Simple CPT dataset preparation - character-based chunking.
Splits text into fixed-size character chunks with overlap.
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm


def chunk_text_simple(text, chunk_size=8192, overlap=256):
    """
    Split text into overlapping chunks based on characters.
    
    Args:
        text: Raw text string
        chunk_size: Size of each chunk in characters (default: 8192)
        overlap: Number of overlapping characters (default: 256)
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move to next chunk with overlap
        start += (chunk_size - overlap)
        
        # Break if we've processed all text
        if end >= text_length:
            break
    
    return chunks


def prepare_cpt_dataset_simple(
    input_file: str,
    output_file: str,
    chunk_size: int = 8192,
    overlap: int = 256,
    max_chunks: int = None,
):
    """
    Prepare CPT dataset from raw text file using simple character-based chunking.
    
    Args:
        input_file: Path to raw text file
        output_file: Path to output JSON file
        chunk_size: Chunk size in characters (default: 8192)
        overlap: Number of overlapping characters (default: 256)
        max_chunks: Maximum number of chunks to create (for testing)
    """
    print(f"\n{'='*60}")
    print("Simple CPT Dataset Preparation")
    print(f"{'='*60}\n")
    
    # Read raw text
    print(f"Reading input file: {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    text_length = len(raw_text)
    print(f"Input file size: {text_length:,} characters")
    
    # Split into chunks
    print(f"\nCreating chunks (size={chunk_size}, overlap={overlap})...")
    chunks = chunk_text_simple(raw_text, chunk_size, overlap)
    
    print(f"Created {len(chunks):,} chunks")
    
    # Limit number of chunks if specified
    if max_chunks and max_chunks < len(chunks):
        print(f"Limiting to {max_chunks:,} chunks for testing")
        chunks = chunks[:max_chunks]
    
    # Create dataset
    print(f"\nCreating dataset...")
    dataset = []
    for idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        sample = {
            "id": f"cpt_{idx:08d}",
            "text": chunk,
            "num_chars": len(chunk),
        }
        dataset.append(sample)
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    print(f"\nSaving dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset):,}")
    print(f"Total characters: {sum(s['num_chars'] for s in dataset):,}")
    print(f"Average chars per sample: {sum(s['num_chars'] for s in dataset) / len(dataset):.0f}")
    print(f"Min chars: {min(s['num_chars'] for s in dataset)}")
    print(f"Max chars: {max(s['num_chars'] for s in dataset)}")
    
    import os
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Output file size: {file_size_mb:.2f} MB")
    print(f"{'='*60}\n")
    
    print(f"✅ Dataset created successfully!")
    print(f"   Use with: --data_path {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CPT dataset with simple character-based chunking"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input raw text file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8192,
        help="Chunk size in characters (default: 8192)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=256,
        help="Number of overlapping characters (default: 256)",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Maximum number of chunks (for testing)",
    )
    
    args = parser.parse_args()
    
    prepare_cpt_dataset_simple(
        input_file=args.input,
        output_file=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()
