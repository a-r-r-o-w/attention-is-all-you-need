from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

with open("requirements.txt", "r", encoding="utf-8") as file:
    requirements = [line.strip() for line in file.readlines() if len(line) > 0 and line[0] != "#"]

setup(
    name="attention_is_all_you_need",
    version="0.0.1",
    description="Simple from-scratch implementation of the reknowned 'Attention Is All You Need' paper.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Aryan V S",
    author_email="contact.aryanvs+attentionisallyouneed@gmail.com",
    url="https://github.com/a-r-r-o-w/attention-is-all-you-need",
    python_requires=">=3.8.0",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=requirements,
    extra_requires={"dev": ["ruff==0.4.3"], "test": ["pytest==8.2.0"]},
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

# Steps to publish:
# 1. Update version in setup.py
# 2. python setup.py sdist bdist_wheel
# 3. Check if everything works with testpypi:
#    twine upload --repository testpypi dist/*
# 4. Upload to pypi:
#    twine upload dist/*
