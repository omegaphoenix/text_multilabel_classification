# Multiclass Classification
Example for using sklearn for multilabel classification.

## Pipenv Setup
```bash
# Pipenv is a python package manager (e.g. pipenv is to Python as npm is to Javascript/Node)
brew install pipenv
# or pip2 install pipenv
# or pip3 install pipenv

# Set this to your standard shell
# For fish, see https://docs.pipenv.org/advanced/#shell-completion
SHELL_RC=~/.bashrc  # or .zshrc

# Enable command line completion
echo 'eval "$(pipenv --completion)"' >> $SHELL_RC

# By default, virtual environments appear under
# ~/.local/share/virtualenvs/python-*
# Use the following to use <PROJECT_ROOT>/.venv instead
echo "export PIPENV_VENV_IN_PROJECT=true" >> $SHELL_RC
```

## Installation and Running
```bash
# Create a virtual environment and install dependent packages in Pipfile
pipenv install

# Open shell in configured virtual environment
pipenv shell

# Once inside the shell, run:
python main.py
```
