import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="parafilt",
    version="0.1.1-beta",
    author="Asaf Zorea",
    author_email="zoreasaf@gmail.com",
    description="ParaFilt is a Python package that provides a collection of parallel adaptive filter "
                "implementations for efficient signal processing applications.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/nuniz/ParaFilt",
    packages=setuptools.find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "tests.*"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch",
    ],
)
