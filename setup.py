import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diver", # Replace with your own username
    version="0.2.1",
    author="Tom Walker",
    author_email='tom.walker@clearscore.com',
    description='diver is a series of tools to speed up common feature-set investigation, conditioning and encoding for common ML algorithms',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ClearScore/diver",
    packages=setuptools.find_packages(),
    install_requires=[
        'scikit-learn>=0.22.1',
        'joblib',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        ],
    python_requires='>=3.6',
)