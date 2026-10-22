from setuptools import find_packages, setup

setup(
    name="stage0_basics",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/stage0_basics"]),
        ("share/stage0_basics", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    maintainer="Kudrat Singh",
    maintainer_email="kudratsingh@users.noreply.github.com",
    description="Week 1 ROS 2 communication exercise",
    license="UNLICENSED",
    entry_points={"console_scripts": [
        "joint_publisher = stage0_basics.joint_publisher:main",
        "joint_subscriber = stage0_basics.joint_subscriber:main",
        "reset_client = stage0_basics.reset_client:main",
    ]},
)
