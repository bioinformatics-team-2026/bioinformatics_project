from collections import namedtuple
from pathlib import Path
from Bio import SeqIO

FastqRead = namedtuple("FastqRead", ["id", "sequence", "quality_scores"])

def parse_fastq(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"FASTQ file not found: {filepath}")
    for record in SeqIO.parse(str(filepath), "fastq"):
        yield FastqRead(
            id=record.id,
            sequence=str(record.seq),
            quality_scores=record.letter_annotations["phred_quality"],
        )

def count_reads(filepath):
    return sum(1 for _ in parse_fastq(filepath=filepath))

def get_read_lengths(filepath):
    return [len(record.seq) for record in parse_fastq(filepath=filepath)]

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fastq_parser.py <fastq_file>")
        sys.exit(1)
    fq_path = sys.argv[1]
    print(f"Parsing: {Path(fq_path).name}")
    total = 0
    for read in parse_fastq(fq_path):
        total += 1
        if total <= 3:
            print(
                f" Read {total}: {read.id} len={len(read.sequence)} "
                f"avg_qual={sum(read.quality_scores)/len(read.quality_scores):.1f}"
            )
    print(f"Total reads: {total}")
