from setuptools import find_packages, setup

setup(
    name="lightning_ir",
    packages=find_packages(
        include=["lightning_ir", "lightning_ir.*"], exclude=["test", "test.*"]
    ),
    entry_points={
        "console_scripts": [
            "lightning-ir = lightning_ir.main:main",
        ]
    },
)