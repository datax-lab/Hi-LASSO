import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hi_lasso",
    version="0.1.0",
    author="Jongkwon Jo",
    author_email="jongkwon.jo@gmail.com",
    description="High-Demensional LASSO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/datax-lab/Hi-LASSO",
    packages=setuptools.find_packages(),
    install_requires=[
          'glmnet', 'tqdm'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)