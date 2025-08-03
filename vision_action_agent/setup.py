from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vision-action-agent",
    version="0.1.0",
    author="Vision Action Team",
    author_email="contact@visionaction.dev",
    description="AI agent for automated screen interaction and PACS workflow automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vision-action-agent",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vision-action=vision_action_agent.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vision_action_agent": [
            "config/*.yaml",
            "templates/*.yaml",
        ],
    },
    keywords="automation, PACS, medical imaging, OCR, computer vision, RPA",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/vision-action-agent/issues",
        "Source": "https://github.com/yourusername/vision-action-agent",
        "Documentation": "https://vision-action-agent.readthedocs.io/",
    },
)