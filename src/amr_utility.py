import numpy as np
# import tqdm
from Bio import SeqIO, Entrez
#from Bio.Alphabet import generic_dna, generic_protein
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
import shutil

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Union
from urllib.parse import urlparse
from urllib.request import urlopen

from pathlib import Path
from typing import List, Tuple, Any
import json
import pandas as pd

from tqdm import tqdm

data_files = {
    "Klebsiella_pneumoniae_aztreonam":{
        "pathogen": "Klebsiella_pneumoniae",
        "antibiotics": "aztreonam",
        "gene": "*",
        "fold": "https://syncandshare.desy.de/public.php/dav/files/XgXfASCFeF2jw4M/folds_Klebsiella_pneumoniae_aztreonam.json",
        "labels": "https://syncandshare.desy.de/public.php/dav/files/XgXfASCFeF2jw4M/labels_Klebsiella_pneumoniae_aztreonam.tsv",
        "seq_raw": "https://syncandshare.desy.de/public.php/dav/files/XgXfASCFeF2jw4M/dlmb_data_Klebsiella_pneumoniae_aztreonam_raw.tar.gz",
        "seq_gene": "https://syncandshare.desy.de/public.php/dav/files/XgXfASCFeF2jw4M/dlmb_data_Klebsiella_pneumoniae_aztreonam_gene.tar.gz",
        "seq_format": "{}.fna"
    }, 
    "Staphylococcus_aureus_cefoxitin":{
        "pathogen": "Staphylococcus_aureus",
        "antibiotics": "cefoxitin",
        "gene": "*",
        "fold": "https://syncandshare.desy.de/public.php/dav/files/XgXfASCFeF2jw4M/folds_Staphylococcus_aureus_cefoxitin.json",
        "labels": "https://syncandshare.desy.de/public.php/dav/files/XgXfASCFeF2jw4M/labels_Staphylococcus_aureus_cefoxitin.tsv",
        "seq_raw": "https://syncandshare.desy.de/public.php/dav/files/XgXfASCFeF2jw4M/dlmb_data_Staphylococcus_aureus_cefoxitin_raw.tar.gz",
        "seq_gene": "https://syncandshare.desy.de/public.php/dav/files/XgXfASCFeF2jw4M/dlmb_data_Staphylococcus_aureus_cefoxitin_gene.tar.gz",
        "seq_format": "{}.fna"
    },
    "Staphylococcus_aureus_cefoxitin_pbp4":{
        "pathogen": "Staphylococcus_aureus",
        "antibiotics": "cefoxitin",
        "gene": "pbp4",
        "fold": "https://syncandshare.desy.de/public.php/dav/files/XgXfASCFeF2jw4M/folds_Staphylococcus_aureus_cefoxitin.json",
        "labels": "https://syncandshare.desy.de/public.php/dav/files/XgXfASCFeF2jw4M/labels_Staphylococcus_aureus_cefoxitin.tsv",
        "seq_raw": "https://syncandshare.desy.de/public.php/dav/files/XgXfASCFeF2jw4M/dlmb_data_Staphylococcus_aureus_cefoxitin_gene_pbp4.tar.gz",
        "seq_gene": "https://syncandshare.desy.de/public.php/dav/files/XgXfASCFeF2jw4M/dlmb_data_Staphylococcus_aureus_cefoxitin_gene_pbp4.tar.gz",
        "seq_format": "{}-pbp4.fna"
    }
}

import tarfile
# … keep the other imports (shutil, tempfile, Path, etc.)

def download_and_extract_zip(
    url: str,
    download_dir: Union[str, Path],
    extract_relative_dir: Union[str, Path],
) -> Path:
    """Download an archive if missing and either extract or copy it into a directory."""
    download_dir_path = Path(download_dir)
    download_dir_path.mkdir(parents=True, exist_ok=True)

    archive_name = Path(urlparse(url).path).name
    if not archive_name:
        raise ValueError(f"Cannot derive a filename from URL: {url}")

    archive_path = download_dir_path / archive_name
    if not archive_path.exists():
        _download_file(url, archive_path)

    extraction_dir = Path(extract_relative_dir)
    extraction_dir.mkdir(parents=True, exist_ok=True)

    suffixes = archive_path.suffixes
    is_tar_gz = suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz"

    if is_tar_gz:
        # print(f"is tar.gz download={download_dir} extract={extract_relative_dir} => extraction_dir={extraction_dir}")
        needs_extract = not any(extraction_dir.iterdir())
        if needs_extract:
            with tarfile.open(archive_path) as tar_file:
                _safe_extract_tar(tar_file, extraction_dir)
        return extraction_dir
    else:
        target_file = extraction_dir / archive_path.name
        if not target_file.exists():
            shutil.copy2(archive_path, target_file)
        # print(f"is file download={download_dir} extract={extract_relative_dir} => target_file={target_file}")
        return target_file


def _download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """Download `url` to `destination` with a progress bar and a temporary file."""
    tmp_path: Path | None = None
    try:
        with urlopen(url) as response:  # nosec B310 - standard library helper
            status = getattr(response, "status", None)
            if status is not None and status >= 400:
                raise RuntimeError(f"Failed to download {url}: HTTP {status}")

            total = response.length or response.getheader("Content-Length")
            try:
                total = int(total)
            except (TypeError, ValueError):
                total = None

            with tempfile.NamedTemporaryFile(
                "wb", delete=False, dir=destination.parent, suffix=".tmp"
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {destination.name}",
                ) as bar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                        bar.update(len(chunk))

        assert tmp_path is not None
        tmp_path.replace(destination)
    except Exception:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        raise

def _safe_extract_tar(tar_file: tarfile.TarFile, destination: Path) -> None:
    """
    Extract `tar_file` into `destination` safely with a progress bar.
    Blocks path traversal (e.g. files trying to escape `destination`).
    """
    members = tar_file.getmembers()

    # Safety check: ensure no member tries to escape destination
    for m in members:
        member_path = Path(m.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe path detected in archive: {m.name}")

    # Progress bar
    with tqdm(total=len(members), unit="file", desc=f"Extracting {destination}") as bar:
        for m in members:
            tar_file.extract(m, path=destination)
            bar.update(1)

# def get_ds_info(pathogen, antibiotics):
#     """
#     returns 
#         "gene": [list of gene names]
#     """
#     return None

def get_seq_label_fold(dataset_name, fold, count, mode="raw"):
    """
    dataset_name:
    gene: gene name or "*" for genome
    fold: from this fold
    count: this number of items

    returns a list of pair of sequence and a label
    """

    # Download seq file
    if mode == "raw":
        sequences_dir = download_and_extract_zip(data_files[dataset_name]["seq_raw"], "download/", f"data/{dataset_name}/seq/")
    elif mode == "gene":
        sequences_dir = download_and_extract_zip(data_files[dataset_name]["seq_gene"], "download/", f"data/{dataset_name}/seq-gene/")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # Download the fold file
    fold_json_path = download_and_extract_zip(data_files[dataset_name]["fold"], "download/", f"data/{dataset_name}/")
    # Download the label file
    labels_csv_path = download_and_extract_zip(data_files[dataset_name]["labels"], "download/", f"data/{dataset_name}/")

    # Load folds JSON
    with open(fold_json_path, "r") as f:
        folds = json.load(f)
    if not isinstance(folds, list) or not all(isinstance(x, list) for x in folds):
        raise ValueError("Fold file must be a JSON list of lists of IDs.")
    if fold < 0 or fold >= len(folds):
        raise IndexError(f"fold_index {fold} out of range [0, {len(folds)-1}].")
    
    # print("Files downloaded and extracted:")

    # IDs for the requested fold (normalize to strings to match filenames and labels)
    fold_ids: List[str] = [str(x) for x in folds[fold]]

    # print("fold_ids:", fold_ids[:10], "..." if len(fold_ids) > 10 else "")


    # Load labels CSV with header: [row_number, id, label]
    df = pd.read_csv(labels_csv_path, sep="\t", header=0, dtype=str)
    if df.shape[1] < 3:
        raise ValueError("Labels file must have at least 3 columns: [row_number, id, label].")

    id_col = df.columns[1]
    label_col = df.columns[2]

    # Build a mapping from id -> label
    labels_map = {row[id_col]: int(row[label_col]) for _, row in df.iterrows()}
    # print(f"Loaded {len(labels_map)} labels from {labels_csv_path}: {"1280.29740" in labels_map}")

    # print("Labels loaded for IDs:", list(labels_map.keys())[:10], "..." if len(labels_map) > 10 else "")

    seq_dir = Path(sequences_dir)
    results: List[Tuple[str, Any]] = []

    # Iterate in the same order as fold_ids
    for i, id_ in enumerate(tqdm(fold_ids, desc=f"Loading fold {fold} of {dataset_name}", leave = False)):
        if count is not None and i >= count:
            break
        # fna_path = seq_dir / f"{id_}.fna"
        fna_path = seq_dir / data_files[dataset_name]["seq_format"].format(id_)
        if not fna_path.exists():
            print("Warning: Sequence file not found for ID:", id_, "file=", fna_path, "format=", data_files[dataset_name]["seq_format"])
            # Skip missing sequences; alternatively raise if strict handling is needed
            # raise FileNotFoundError(f"Sequence file not found: {fna_path}")
            continue
        if id_ not in labels_map:
            # Skip missing labels; alternatively raise if strict handling is needed
            print(f"Warning: Label not found for ID:'{id_}' {labels_map.keys()}")
            # raise KeyError(f"Label not found for ID: {id_}")
            continue

        seqs = {}
        for record in SeqIO.parse(fna_path, "fasta"):
            seq = str(record.seq).upper()
            seqs[record.id] = seq
        # assert(len(seq) > 0)
        if len(seqs) > 0:
            results.append((seqs, labels_map[id_]))
    
    # print(f"data loaded {len(results)} items from fold {fold} of {dataset_name} (requested count={count})")
    return results


def get_seq_label_hard(dataset_name, mode = "raw"):
    test = get_seq_label_fold(dataset_name, 0, None, mode=mode)
    train = []
    for i in range(1, 9):
        train.extend(get_seq_label_fold(dataset_name, i, None, mode=mode))
    # print(f"data loaded {len(train)} train and {len(test)} test items")

    return {"train": train, "test": test}


def get_seq_label_simple(dataset_name):
    # x = get_seq_label_fold(dataset_name, 0, 100)
    # print([(seqs, ll) for seqs, ll in x if len(seqs) != 1])
    test = [(next(iter(s.values())), ll) for s, ll in get_seq_label_fold(dataset_name, 0, 100)]
    train = []
    for i in range(1, 9):
        for seqs, label in get_seq_label_fold(dataset_name, i, 100):
            assert(len(seqs) == 1)
            train.append((next(iter(seqs.values())), label))

    return {"train": train, "test": test}



# Example:
# seq_label_Kp_Az_pbp4_train = get_seq_label("Klebsiella_pneumoniae", "aztreonam", "train", "pbp4")
# seq_label_Kp_Az_pbp4_test = get_seq_label("Klebsiella_pneumoniae", "aztreonam", "test", "pbp4")
# seq_train = [x[0] for x in seq_label_Kp_Az_pbp4_train]
# y_train = [x[1] for x in seq_label_Kp_Az_pbp4_train]
# seq_test = [x[0] for x in seq_label_Kp_Az_pbp4_test]
# y_test = [x[1] for x in seq_label_Kp_Az_pbp4_test]


# data_files = {
#     "Klebsiella_pneumoniae_aztreonam":{
#         "train_seq": "data/Klebsiella_pneumoniae_aztreonam/train_seq.txt",
#         "test_label": "data/Klebsiella_pneumoniae_aztreonam/test_label.txt",
#         "test_seq": "data/Klebsiella_pneumoniae_aztreonam/test_seq.txt",
#         "train_label": "data/Klebsiella_pneumoniae_aztreonam/train_label.txt"
#     }, 
#     "Staphylococcus_aureus_cefoxitin":{
#         "train_seq": "data/Staphylococcus_aureus_cefoxitin/train_seq.txt",
#         "test_label": "data/Staphylococcus_aureus_cefoxitin/test_label.txt",
#         "test_seq": "data/Staphylococcus_aureus_cefoxitin/test_seq.txt",
#         "train_label": "data/Staphylococcus_aureus_cefoxitin/train_label.txt"
#     }
# }

# def wc(fn):
#     return len(open(fn).readlines())

# def gene_with_high_incidence(gene_wc, wc):
#     return [g for g, c in gene_wc.items() if c >= wc * 1.0]


# def create_gene_datasets(prefix_data_folder, output_data_folder):
#     # shutil.rmtree('../data/ds1')
#     # if os.path.exists(output_data_folder) and os.path.isdir(output_data_folder):
#     #     shutil.rmtree(output_data_folder)
#     os.makedirs(output_data_folder, exist_ok=True)


#     for ds_name, ds_values in data_files.items():
#         # print(ds_name)
#         gene_sequences_tt = {}
#         wc_tt = {}
#         gene_wc_tt = {}
#         for var_seq_name, var_label_name, var_dest_folder in [("train_seq", "train_label", "train"), ("test_seq", "test_label", "test")]:
#             gene_sequences = {}
#             for cur_record in SeqIO.parse(prefix_data_folder + ds_values[var_seq_name], "fasta"):
#                 seq_name_gene = cur_record.name.split(';')[0]
#                 seq_name, seq_gene = seq_name_gene.split('_')
#                 # print(seq_gene)
#                 if seq_gene == '': continue
#                 if seq_gene not in gene_sequences: gene_sequences[seq_gene] = {}
#                 gene_sequences[seq_gene][seq_name] = cur_record.seq
#             gene_sequences_tt[var_dest_folder] = gene_sequences
#             wc_tt[var_dest_folder] = wc(prefix_data_folder + ds_values[var_label_name])
#             gene_wc_tt[var_dest_folder] = {seq_gene:len(seq_name_seq) for seq_gene, seq_name_seq in gene_sequences.items()}

#         gene_rich_set = set(gene_with_high_incidence(gene_wc_tt["test"], wc_tt["test"])) & set(gene_with_high_incidence(gene_wc_tt["train"], wc_tt["train"]))
#         # for gr in gene_rich_set:
#         #     print(gr, gene_wc_tt["train"][gr], wc_tt["train"], gene_wc_tt["test"][gr], wc_tt["test"])
#         for var_seq_name, var_label_name, var_dest_folder in [("train_seq", "train_label", "train"), ("test_seq", "test_label", "test")]:
#             gene_sequences = gene_sequences_tt[var_dest_folder]
#             newpath = output_data_folder + "/" + ds_name + "/" + var_dest_folder + "/"
#             if not os.path.exists(newpath):
#                 os.makedirs(newpath)
#             shutil.copyfile(prefix_data_folder + ds_values[var_label_name], newpath + "labels.txt")

#             for gene_name, seq_name_seq in gene_sequences.items():
#                 if gene_name in gene_rich_set:
#                     with open(newpath + gene_name + ".fasta", "w") as f:
#                         for n, s in seq_name_seq.items():
#                             print(">" + n + "\n" + s, file = f)

# # Helper function for loading gene sequences
# def load_gene_data(folder, dataset, gene):
#     '''
#     loads genemic sequences and labels for train and test sets for a specific dataset and gene.
#     Example:
#       ds = load_gene_data("../data/ds1", "Klebsiella_pneumoniae_aztreonam", "acrR")
#     here ds["train"] and ds["test"] both are a list of tuples of the form (gene, seq, label).

#     '''
#     # folder = "../data/ds1"
#     # dataset = "Klebsiella_pneumoniae_aztreonam"
#     # gene = "acrR"

#     pathogens = {}
#     for tt in ["train", "test"]:
#         pathogen_name_to_seq, pathogen_name_to_label = {}, {}
#         for cur_record in SeqIO.parse(folder + "/" + dataset + "/" + tt + "/" + gene + ".fasta", "fasta"):
#             pathogen_name_to_seq[cur_record.name] = str(cur_record.seq)

#         for l in open(folder + "/" + dataset + "/" + tt + "/" + "labels" + ".txt"):
#             x = l.strip().split('\t')
#             pathogen_name_to_label[x[0]] = int(x[1])

#         pathogens_tt = []
#         for g, seq in pathogen_name_to_seq.items():
#             if g in pathogen_name_to_label:
#                 pathogens_tt.append((g, seq.upper(), pathogen_name_to_label[g]))
#         pathogens[tt] = pathogens_tt

#     # print(pathogens)
#     return pathogens

