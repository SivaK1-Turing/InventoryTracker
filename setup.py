# setup.py
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import os
import sys
import subprocess

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        self.update_documentation()
    
    def update_documentation(self):
        """Update documentation from source code."""
        script_path = os.path.join('scripts', 'update_docs.py')
        if os.path.exists(script_path):
            subprocess.run([sys.executable, script_path], check=True)
            print("Documentation updated.")

setup(
    name="inventorytracker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "typer>=0.4.0",
        "pydantic>=1.8.0",
        "rich>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "invtrack=inventorytracker.main:app",
            "update-docs=scripts.update_docs:main",
        ],
        "codesnip_product_hooks": [],
    },
    cmdclass={
        'develop': PostDevelopCommand,
    },
)