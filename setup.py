import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="shapwaterfall",     
    version="0.1.4",
    author="John Halstead",
    author_email="jhalstead@vmware.com",
    description="A SHAP Waterfall Chart interpreting local differences between observations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JohnBHalstead/shapwaterfall",
    install_requires=[            
          'shap',
          'waterfallcharts',
          'numpy',
          'matplotlib',
          'pandas'
      ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
