import pkg_resources

# List of packages used in your script
packages = [
    "streamlit",
    "pandas",
    "seaborn",
    "matplotlib",
    "streamlit-option-menu",
    "Pillow",
      # Optional: standard pickle does not require an external library
]

# Create or overwrite the requirements.txt file
with open("requirements.txt", "w") as file:
    for package in packages:
        try:
            # Get the package version
            version = pkg_resources.get_distribution(package).version
            # Write package with version into requirements.txt
            file.write(f"{package}=={version}\n")
        except pkg_resources.DistributionNotFound:
            print(f"Package '{package}' is not installed.")
            # Optionally, write the package name without a version
            file.write(f"{package}\n")

print("Requirements file has been created!")
