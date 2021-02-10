import pathlib
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--supplementary_pdf", type=str)
args = parser.parse_args()
supplementary = pathlib.Path(args.supplementary_pdf) if args.supplementary_pdf else None
assert supplementary.exists() and supplementary.is_file()

root = (pathlib.Path(__file__).parent / "..")

ignore_file_extensions = ["png", "gif"]
forbidden_words = ["jb", "cordonnier", "loukas", "jaggi", "epfml", "epfl", "mlbench", "epfml", "github", "arxiv", "eprint"]
filter_files = [".git", "tools", "__pycache__", ".DS_Store", ".zip"]


def copy_anonymized(source_file, target_file):
    with open(source_file, "r") as inf, open(target_file, "w") as outf:
        for line in inf:
            if any(forbidden.lower() in line.lower() for forbidden in forbidden_words):
                outf.write("#- anonymized\n")
            else:
                outf.write(line)

try:
    tmp_directory = root / "tmp"
    if tmp_directory.exists():
        shutil.rmtree(tmp_directory)

    tmp_directory.mkdir()

    files = [f for f in root.glob("**/*") if not any(s in str(f) for s in filter_files)]
    print(root, files)

    for source_file in files:
        target_file = tmp_directory / str(source_file)[3:]
        print(f"{source_file} -> {target_file}")
        if source_file.is_dir():
            continue
        target_file.parent.mkdir(parents=True, exist_ok=True)

        # Simply copy if not a text file
        if any(str(source_file).endswith(ext) for ext in ignore_file_extensions):
            shutil.copy(source_file, target_file)
        # Otherwise anonymise the file line by line
        else:
            copy_anonymized(source_file, target_file)

    # If we provide a PDF, we move the content of tmp/ into tmp/code/ and add the PDF
    if supplementary:
        tmp_directory.rename(root / "code")
        tmp_directory.mkdir()
        (root / "code").rename(tmp_directory / "code")
        shutil.copy(supplementary, tmp_directory / "supplementary.pdf")
        print(f"Added PDF file '{supplementary}'.")

    # Make the zip archive.
    zip_file = (root / "supplementary").absolute().resolve()
    shutil.make_archive(zip_file, 'zip', tmp_directory)
    print(f"Anonymous supplementary archive saved in '{zip_file}.zip'")

finally:
    # Clean up
    shutil.rmtree(tmp_directory)
