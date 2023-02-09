import setuptools

# requirements = open("requirements36.txt").read().splitlines()
requirements = open("requirements38.txt").read().splitlines()
# dev_requirements = open("requirements_dev.txt").read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GDL",
    description="Official release of code accompanying EMOCA, CVPR2022",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Radek Danecek",
    author_email="danekradek@gmail.com",
    version="0.0.3",
    packages=["gdl", "gdl_apps"],
    package_dir={"": "."},
    # install_requires=requirements,
    # extras_require={"dev": dev_requirements},
    # python_requires=">=3.6",
    python_requires=">=3.8",
    
)