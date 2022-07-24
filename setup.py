from setuptools import setup, find_packages

base_packages = ["scikit-learn>=0.24.2"]

dev_packages = ["pytest>=4.0.2"]

setup(
    name="drosophila",
    version="0.0.1",
    license="MIT",
    description="Fruit Fly Algorithm For Embeddings",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.5",
    author="Amit Chaudhary",
    author_email="meamitkc@gmail.com",
    url="https://github.com/amitness/drosophila",
    install_requires=base_packages,
    extras_require={"dev": dev_packages},
    packages=find_packages(),
)
