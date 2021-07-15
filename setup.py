import setuptools

requirements = open("requirements.txt").read().splitlines()
# dev_requirements = open("requirements_dev.txt").read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GDL",
    description="Radek's messy library for deep learning gdl_apps. Mostly with focus on 2D to 3D DL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Radek Danecek",
    author_email="danekradek@gmail.com",
    version="0.0.2",
    packages=["gdl", "gdl_apps"],
    package_dir={"": "."},
    install_requires=requirements,
    # extras_require={"dev": dev_requirements},
    python_requires=">=3.6",
)