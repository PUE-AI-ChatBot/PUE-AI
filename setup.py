import setuptools

setuptools.setup(
    name="chatbot-pue",
    version="1.0.0",
    license='LGPL-2.1',
    author="PUE",
    author_email="well87865@gmail.com",
    description="This is a simple chatbot package, only for korean.",
    long_description=open('README.md').read(),
    url="https://github.com/PUE-AI-ChatBot/PUE-AI",
    packages=setuptools.find_packages(),
    classifiers=[
        # 패키지에 대한 태그
        "Programming Language :: Python :: 3.9",
        "License :: LGPL-2.1 License",
        "Operating System :: OS Independent"
    ],
)
