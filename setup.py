from setuptools import find_packages, setup

setup(
    name="aprr",
    version="1.0.0",
    description="Adaptive Probabilistic Routing Reinforcement for multi-agent LLM workflows.",
    author="Jenisha T",
    author_email="joyjeni@gmail.com",
    url="https://github.com/joyjeni/aprr-multi-agent-routing",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "scipy>=1.10",
    ],
    extras_require={
        "dev": ["pytest>=7", "ruff", "black"],
        "viz": ["seaborn>=0.13"],
    },
)
