from setuptools import setup, find_packages

setup(
    name="llm-cli-tool",
    version="0.1.0",
    author="fatSheep",
    author_email="tzh2005t@163.com",
    description="A command-line tool for interacting with large language models.\nOnly the platform SiliconFlow is supported now",
    packages=find_packages(),
    install_requires=[
        "requests",
        "argparse",
    ],
    entry_points={
        "console_scripts": [
            "llm=src.cli:main",
            "llm-cleanup=src.uninstall:main",
            "llm-analyze=src.log_analyzer:main",
        ],
    },
    python_requires=">=3.6",
)
