from setuptools import setup, find_packages

setup(name="MetaRecognition", version="1.0", author="TRAN DAI CHI", author_email="ctran743@gmail.com", description="README.md", url="",
      py_modules=["dataloader", "dataset", "models"], license="LICENSE", python_requires=">=3.8", include_package_data=True, install_requires="requirements.txt")