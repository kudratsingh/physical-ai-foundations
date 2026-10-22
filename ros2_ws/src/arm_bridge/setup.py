from setuptools import find_packages, setup

setup(
    name="arm_bridge",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/arm_bridge"]),
        ("share/arm_bridge", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    maintainer="Kudrat Singh",
    maintainer_email="kudratsingh@users.noreply.github.com",
    license="UNLICENSED",
    entry_points={
        "console_scripts": [
            "arm_simulator = arm_bridge.simulator:main",
            "arm_controller = arm_bridge.controller:main",
            "arm_dashboard = arm_bridge.dashboard:main",
        ]
    },
)
