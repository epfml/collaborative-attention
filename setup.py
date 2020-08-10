import setuptools

setuptools.setup(
    name="collaborative-attention",
    version="0.1.0",
    author="Jean-Baptiste Cordonnier",
    author_email="jean-baptiste.cordonnier@epfl.ch",
    description="",
    url="https://github.com/collaborative-attention",
    packages=["collaborative_attention"],
    package_dir={"": "src/"},
    python_requires=">=3.6",
    install_requires=[
        "tensorly>=0.4.5",
        "transformers==2.11.0",
        "parameterized>=0.7.4",
        "tqdm>=4.46.0",
        "wandb==0.9.2",
    ],
)
