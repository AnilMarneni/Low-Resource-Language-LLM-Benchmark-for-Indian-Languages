from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="llm_indic_benchmark",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements(),
    description="Benchmarking Framework for LLMs on Low-Resource Indian Languages",
    author="AI Researcher",
)
