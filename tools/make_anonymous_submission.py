import pathlib
import shutil


root = pathlib.Path(__file__).parent / ".."

ignore_file_extensions = ["png", "gif"]

def should_be_anonymized(line):
    return any(
        (s in line.lower())
        for s in ["jb", "cordonnier", "loukas", "jaggi", "epfml", "epfl", "mlbench", "epfml", "github", "arxiv", "eprint"]
    )

try:
    tmp_directory = root / "tmp"
    if tmp_directory.exists():
        shutil.rmtree(tmp_directory)

    tmp_directory.mkdir()

    print(tmp_directory, root)

    filter_files = [".git", "tools", "__pycache__", ".DS_Store", ".zip"]
    files = [f for f in root.glob("**/*") if not any(s in str(f) for s in filter_files)]

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
            with open(source_file, "r") as inf:
                with open(target_file, "w") as outf:
                    for line in inf:
                        if should_be_anonymized(line):
                            outf.write("#- anonymized\n")
                        else:
                            outf.write(line)

    # Make the zip archive.
    zip_file = (root / "code").absolute().resolve()
    shutil.make_archive(zip_file, 'zip', tmp_directory)
    print(f"Anonymous code archive saved in '{zip_file}.zip'")

finally:
    # Clean up
    shutil.rmtree(tmp_directory)
