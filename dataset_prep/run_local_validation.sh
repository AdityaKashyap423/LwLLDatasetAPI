#!/bin/bash

mypy . && echo "mypy passed!"
flake8 && echo "flake8 passed!"
echo "All validation passed!"